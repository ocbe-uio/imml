

class Utils:

    @staticmethod
    def convert_df_to_r_object(dataframe):
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri

        base = importr('base')
        with (ro.default_converter + pandas2ri.converter).context():
            r_from_pd_df = ro.conversion.get_conversion().py2rpy(dataframe)
        return base.lapply(r_from_pd_df, base.as_matrix)
