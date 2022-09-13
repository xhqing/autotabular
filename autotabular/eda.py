
class AutoEDA(object):

    def get_info(self, df):
        eda_info = {}
        eda_info['cat_cols'], eda_info['num_cols'] = self.recognize_col2type(df)
        return eda_info

    def recognize_col2type(self, df):
        m, n = df.shape
        cat_cols = []
        num_cols = []
        if n > 1000:
            num_cols = ['n_{}'.format(col) for col in df.columns]
            df.columns = num_cols
        else:
            for col in df.columns:
                nunique = df[col].nunique()
                min_v = df[col].min()
                if nunique == 1:
                    df.drop(col, axis=1, inplace=True)
                else:
                    if nunique < 30 and (min_v == 0 or min_v == 1):
                        col_name = 'c_{}'.format(col)
                        cat_cols.append(col_name)
                        df.rename(columns={col: col_name}, inplace=True)
                    else:
                        col_name = 'n_{}'.format(col)
                        num_cols.append(col_name)
                        df.rename(columns={col: col_name}, inplace=True)
        return cat_cols, num_cols