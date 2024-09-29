def create_total_services_feature(df):
    df['TotalServices'] = df[['PhoneService', 'InternetService_Fiber optic', 'InternetService_DSL']].sum(axis=1)
    return df
