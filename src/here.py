from data_preprocessing import preprocess_mask_data, preprocess_age_data

mask_train, mask_val, mask_test = preprocess_mask_data()
age_train, age_val, age_test = preprocess_age_data(limit=5000)