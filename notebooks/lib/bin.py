from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, hour, dayofweek, unix_timestamp, when, avg, lit,
    max as spark_max, min as spark_min, row_number, lead, stddev
)
from pyspark.sql.window import Window
from pyspark.ml import PipelineModel
from datetime import datetime, timezone


class BinFullPredictor:
    def __init__(self, model_path='./models/RegressionModel'):
        """
        Initializes the BinFullPredictor.

        Parameters:
        - model_path: Path to the saved regression model.
        """
        # Initialize Spark session
        self.spark = SparkSession.builder.appName("SmartLitter").getOrCreate()

        # Database connection properties
        jdbc_url = "jdbc:postgresql://db:5432/litter_db"
        connection_properties = {
            "user": "root",
            "password": "pwd123",
            "driver": "org.postgresql.Driver",
            "stringtype": "unspecified"
        }

        self.litter_bin_geoposition_df = self.spark.read.jdbc(
            url=jdbc_url,
            table="litter_bin_geoposition",
            properties=connection_properties
        )

        # Load bin_state_notification table once
        self.bin_state_df = self.spark.read.jdbc(
            url=jdbc_url,
            table='public.bin_state_notification',
            properties=connection_properties
        ).select('litter_bin_uuid', 'point_in_time', 'notification_type')
        self.bin_state_df.createOrReplaceTempView("notification")

        # Convert 'point_in_time' to timestamp
        self.bin_state_df = self.bin_state_df.withColumn('point_in_time', to_timestamp('point_in_time'))

        # Load the trained pipeline model
        self.model = PipelineModel.load(model_path)

        # Cache the data to improve performance
        self.bin_state_df.cache()

    def get_all_bin_predictions(self, delta_hours):
        """
        Predicts whether bins will be full after delta_hours from now, considering the time since they were last emptied.

        Parameters:
        - delta_hours: The number of hours in the future to predict bin fullness.

        Returns:
        - A Spark DataFrame with columns 'litter_bin_uuid' and 'is_full'.
        """
        # Get the latest 'emptied' notification for each bin
        empty_bins_df = self.spark.sql("""
            SELECT litter_bin_uuid, notification_type, point_in_time FROM (
                SELECT
                    notification.*,
                    ROW_NUMBER() OVER (PARTITION BY litter_bin_uuid ORDER BY point_in_time DESC) AS row_num
                FROM notification
            ) AS ranked
            WHERE row_num = 1 AND notification_type = 'emptied';
        """)

        # Get the current time in UTC
        current_time = datetime.now(timezone.utc)

        # Calculate the duration since the last 'emptied' event for each bin
        empty_bins_df = empty_bins_df.withColumn(
            "duration_since_emptied_hours",
            (unix_timestamp(lit(current_time)) - unix_timestamp(col("point_in_time"))) / 3600.0  # in hours
        )

        # Create bin_durations_df with 'litter_bin_uuid' and 'duration_since_emptied_hours'
        bin_durations_df = empty_bins_df.select("litter_bin_uuid", "duration_since_emptied_hours")

        # Extract list of bin UUIDs
        bin_uuids = [row['litter_bin_uuid'] for row in bin_durations_df.select('litter_bin_uuid').distinct().collect()]

        # Filter events for the specified bin_uuids
        bin_events_df = self.bin_state_df.filter(col('litter_bin_uuid').isin(bin_uuids))

        # Check if we have data for the bins
        bins_with_data = bin_events_df.select('litter_bin_uuid').distinct()
        bins_with_data_list = [row['litter_bin_uuid'] for row in bins_with_data.collect()]

        if not bins_with_data_list:
            print("No data found for the specified bins.")
            return self.spark.createDataFrame([], schema='litter_bin_uuid string, is_full boolean')

        # Define window specifications
        window_spec_desc = Window.partitionBy('litter_bin_uuid').orderBy(col('point_in_time').desc())

        # Create flags for 'emptied' and 'full' events
        bin_events_df = bin_events_df.withColumn('is_emptied',
                                                 when(col('notification_type') == 'emptied', 1).otherwise(0))
        bin_events_df = bin_events_df.withColumn('is_full', when(col('notification_type') == 'full', 1).otherwise(0))

        # Get the latest event for each bin
        latest_events_df = bin_events_df.withColumn('rn', row_number().over(window_spec_desc)) \
            .filter(col('rn') == 1) \
            .select('litter_bin_uuid', 'point_in_time') \
            .withColumnRenamed('point_in_time', 'current_time')

        # Use current time in UTC
        current_time = datetime.now(timezone.utc)
        hour_of_day = current_time.hour
        day_of_week = current_time.isoweekday()  # 1 (Monday) through 7 (Sunday)

        # Prepare bins DataFrame with current time features
        bins_df = latest_events_df.withColumn('current_time', lit(current_time)) \
            .withColumn('hour_of_day', lit(hour_of_day)) \
            .withColumn('day_of_week', lit(day_of_week))

        # Calculate previous 'time_to_full_hours' per bin
        # Get the last 'emptied' event before the current time per bin
        last_emptied_df = bin_events_df.filter(col('is_emptied') == 1) \
            .groupBy('litter_bin_uuid') \
            .agg(spark_max('point_in_time').alias('last_emptied_time'))

        # Get the next 'full' event after the last 'emptied' event per bin
        next_full_df = bin_events_df.alias('events') \
            .join(last_emptied_df, on='litter_bin_uuid', how='inner') \
            .filter((col('events.is_full') == 1) & (col('events.point_in_time') > col('last_emptied_time'))) \
            .groupBy('litter_bin_uuid') \
            .agg(spark_min('events.point_in_time').alias('next_full_time'))

        # Calculate previous time to full in hours
        prev_time_to_full_df = last_emptied_df.join(next_full_df, on='litter_bin_uuid', how='left') \
            .withColumn('prev_time_to_full_hours',
                        (unix_timestamp('next_full_time') - unix_timestamp('last_emptied_time')) / 3600)

        # Calculate average and standard deviation time difference per bin
        # Step 1: For each 'emptied' event, find the next 'full' event
        emptied_events_df = bin_events_df.filter(col('is_emptied') == 1)
        full_events_df = bin_events_df.filter(col('is_full') == 1)

        # Combine 'emptied' and 'full' events
        events_df = emptied_events_df.select('litter_bin_uuid', 'point_in_time', lit('emptied').alias('event_type')) \
            .union(full_events_df.select('litter_bin_uuid', 'point_in_time', lit('full').alias('event_type')))

        # Order events and use lead to find next 'full' event after 'emptied'
        window_spec = Window.partitionBy('litter_bin_uuid').orderBy('point_in_time')
        events_df = events_df.withColumn('next_event_type', lead('event_type').over(window_spec)) \
            .withColumn('next_point_in_time', lead('point_in_time').over(window_spec))

        # Filter to rows where event is 'emptied' and next event is 'full'
        time_diffs_df = events_df.filter((col('event_type') == 'emptied') & (col('next_event_type') == 'full')) \
            .withColumn('time_diff_hours',
                        (unix_timestamp('next_point_in_time') - unix_timestamp('point_in_time')) / 3600)

        # Compute average and standard deviation time difference per bin
        stats_time_to_full_df = time_diffs_df.groupBy('litter_bin_uuid') \
            .agg(
            avg('time_diff_hours').alias('avg_time_to_full_hours'),
            stddev('time_diff_hours').alias('stddev_time_to_full_hours')
        )

        # Combine previous time to full with average and stddev time to full
        time_to_full_df = prev_time_to_full_df.join(stats_time_to_full_df, on='litter_bin_uuid', how='left') \
            .withColumn('prev_time_to_full_hours',
                        when(col('prev_time_to_full_hours').isNull(), col('avg_time_to_full_hours'))
                        .otherwise(col('prev_time_to_full_hours'))) \
            .select('litter_bin_uuid', 'prev_time_to_full_hours', 'avg_time_to_full_hours', 'stddev_time_to_full_hours')

        # Handle bins without previous data by using overall averages
        overall_stats = stats_time_to_full_df.agg(
            avg('avg_time_to_full_hours').alias('overall_avg_time_to_full'),
            avg('stddev_time_to_full_hours').alias('overall_stddev_time_to_full')
        ).collect()[0]

        overall_avg_time_to_full = overall_stats['overall_avg_time_to_full']
        overall_stddev_time_to_full = overall_stats['overall_stddev_time_to_full']

        feature_df = bins_df.join(time_to_full_df, on='litter_bin_uuid', how='left')
        feature_df = feature_df.fillna({
            'prev_time_to_full_hours': overall_avg_time_to_full,
            'avg_time_to_full_hours': overall_avg_time_to_full,
            'stddev_time_to_full_hours': overall_stddev_time_to_full
        })

        # Join with bin_durations_df to get duration_since_emptied_hours
        feature_df = feature_df.join(bin_durations_df, on='litter_bin_uuid', how='left')

        # Make predictions
        predictions_df = self.model.transform(feature_df)

        # Calculate total future hours: duration_since_emptied_hours + delta_hours
        total_future_hours_col = col('duration_since_emptied_hours') + lit(delta_hours)

        # Determine if bins will be full after the total_future_hours
        predictions_df = predictions_df.withColumn('is_full', col('prediction') <= total_future_hours_col)

        # Select relevant columns
        result_df = predictions_df.select('litter_bin_uuid', 'is_full')

        return result_df
