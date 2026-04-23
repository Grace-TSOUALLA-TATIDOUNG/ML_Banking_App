import data_preprocessing as dp

from ml_pipeline import data_split, cross_validation, define_models, model_selection_and_training, model_evaluation


if __name__ == "__main__" :

    bank_df = dp.read_data("data/bank-additional-full.csv")

    bank_df = dp.remove_useless_columns(bank_df)

    bank_df = dp.feature_engineering(bank_df)

    # bank_df = dp.group_data(bank_df)

    bank_df = dp.data_types_review(bank_df)

    bank_df = dp.data_encoding(bank_df)

    X_train, X_test, y_train, y_test = data_split(bank_df)
    
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    scale_pos_weight = neg / pos

    cv_model = cross_validation(number_splits=5)

    models_params = define_models(scale_pos_weight)

    best_model_name, best_params, best_model = model_selection_and_training(models_params, X_train, y_train, cv_model)

    print(f"best_model_name : {best_model_name} \n\n")
    print(f"params : {best_params} \n\n")
    print(f"best_model : {best_model} \n\n")

    model_evaluation(best_model, X_test, y_test)


