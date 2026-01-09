import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc,
    accuracy_score
)
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor




st.set_page_config(page_title="Interactive Data Analysis Platform", layout="wide")

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] button {
        height: 70px;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



PARAM_GRIDS = {
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"]
    },
    "KNN": {
        "model__n_neighbors": [3, 5, 7, 9],
        "model__weights": ["uniform", "distance"]
    },
    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5]
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 150],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3]
    },
    "XGBoost Regressor": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0]
    },
    "Random Forest Regressor": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5]
    },
    "Gradient Boosting Regressor": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5]
    }
}




def detect_leakage_features(df, target_col):

    leakage_candidates = set() 
    suspicious_keywords = [
        "rate", "spread", "charge", "approved", "status",
        "decision", "final", "after", "outcome"
    ]

    for col in df.columns:
        if col == target_col:
            continue

        if any(k in col.lower() for k in suspicious_keywords):
            leakage_candidates.add(col)

    return list(leakage_candidates)




def evaluate_model(fitted_pipeline, X_test, y_test, problem_type):

    y_pred = fitted_pipeline.predict(X_test)

    if problem_type == "classification":

        if hasattr(fitted_pipeline.named_steps["model"], "predict_proba"):
            y_proba = fitted_pipeline.predict_proba(X_test)[:, 1]
        else:
            y_proba = None

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred)
        }


        if y_proba is not None:
            metrics["ROC AUC"] = roc_auc_score(y_test, y_proba)
            metrics["Average Precision"] = average_precision_score(y_test, y_proba)

        return metrics

    else:  # regression

        return {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred)
        }


def run_training_pipeline(df, run_config):

    df = df.copy()

    target_col = run_config["problem"]["target_col"]

    if target_col not in df.columns:
        raise ValueError(
            f"The target column is not in the dataset: {target_col}. "
    )

    auto_leaks = detect_leakage_features(df, target_col)


    final_drops = list(
        (
            set(run_config["feature_drop"]["columns"]) |
            set(auto_leaks)
        ) - {target_col}
    )

    df = df.drop(columns=final_drops, errors="ignore")

    X = df.drop(columns=[target_col])
    y = df[target_col]


    label_encoder = None
    if y.dtype == "object" or y.dtype.name == "category":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)


    st.session_state["label_encoder"] = label_encoder


    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()


    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features)
    ])


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=run_config["split"]["test_size"],
        random_state=run_config["split"]["random_state"]
    )


    model_name = run_config["model"]["name"]
    problem_type = run_config["problem"]["problem_type"]

    if problem_type == "classification":
        if model_name == "KNN":
            model = KNeighborsClassifier()
        elif model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif model_name == "Gradient Boosting":
            model = GradientBoostingClassifier()
        else:
            raise ValueError("Unknown model")

    else:  # regression
        if model_name == "Multiple Linear Regression":
            model = LinearRegression()

        elif model_name == "Random Forest Regressor":
            model = RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )

        elif model_name == "Gradient Boosting Regressor":
            model = GradientBoostingRegressor(
                random_state=42
            )

        elif model_name == "XGBoost Regressor":
            model = XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )


    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])


    if run_config["training"]["strategy"] == "default": #default parameters
        pipeline.fit(X_train, y_train)

    else:  # hyperparameter tuning

        param_grid = PARAM_GRIDS.get(model_name)

        if param_grid is None:
            raise ValueError(f"Tuning parameter is not defined: {model_name}")

        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="roc_auc" if problem_type == "classification" else "r2",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        pipeline = grid.best_estimator_

        st.info(f"Best params: {grid.best_params_}")


    st.session_state["pipeline"] = pipeline 
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test

    return {
        "status": "success",
        "model": model_name,
        "problem_type": problem_type,
        "train_size": X_train.shape,
        "test_size": X_test.shape
    }


if "active_page" not in st.session_state:
    st.session_state.active_page = None

if "df" not in st.session_state:
    st.session_state.df = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

if "target_col" not in st.session_state:
    st.session_state.target_col = None 

if "problem_model" not in st.session_state:
    st.session_state.problem_model = "Regression"


if "encoding_config" not in st.session_state:
    st.session_state["encoding_config"] = {}


if "dropped_features" not in st.session_state:
    st.session_state.dropped_features = []



st.sidebar.title("Pipeline")



if st.sidebar.button("Data", use_container_width=True):
    st.session_state.active_page = "data"


if st.sidebar.button("Target & Problem", use_container_width=True):
    st.session_state.active_page = "target"


if st.sidebar.button("Validation", use_container_width=True):
    st.session_state.active_page = "validation"


if st.sidebar.button("Preprocessing", use_container_width=True):
    st.session_state.active_page = "preprocessing"


if st.sidebar.button("Modeling", use_container_width=True):
    st.session_state.active_page = "modeling"


if st.sidebar.button("Evaluation", use_container_width=True):
    st.session_state.active_page = "evaluation"


if st.sidebar.button("Prediction", use_container_width=True):
    st.session_state.active_page = "prediction"





page = st.session_state.active_page
df = st.session_state.df
target_col = st.session_state.target_col


if page == "data":
    st.header("Data Page")

    if st.button("Clear dataset"):
        st.session_state.df = None
        st.session_state.uploaded_name = None
        if "uploader" in st.session_state:
            del st.session_state["uploader"]
        st.rerun()

    uploaded_file = st.file_uploader(
        "Upload your input CSV file",
        type=["csv"],
        key="uploader"
    )

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_name = uploaded_file.name
        df = st.session_state.df

    st.divider()
    
    if df is not None:
        st.subheader("Basic Info")
        st.write("Uploaded file:", st.session_state.uploaded_name)
        st.write("Number of rows:", df.shape[0])
        st.write("Number of columns:", df.shape[1])

        st.divider()
        st.subheader("Preview")
        st.write(df.head())

        st.divider()
        st.header("Schema summary")
        st.write(df.dtypes)

        st.divider()
        st.header("Quality Overview")
        st.write("Number of missing values: ", df.isna().sum().sum())
        st.write("Number of duplicates: ", df.duplicated().sum())

    else:
        st.subheader("Basic Info")
        st.info("Please upload a CSV file.")

        st.divider()
        st.subheader("Preview")
        st.info("Please upload a CSV file.")

        st.divider()
        st.header("Schema summary")
        st.info("Please upload a CSV file.")

        st.divider()
        st.header("Quality Overview")
        st.info("Please upload a CSV file.")



elif page == "target":
    st.title("Target & Problem Page")

    if df is None:
        st.warning("First, load the dataset from the Data page.")
    else:
        st.subheader("Target Selection")

        if st.session_state.target_col in df.columns:
            default_index = df.columns.get_loc(st.session_state.target_col)
        else:
            default_index = 0

        st.session_state.target_col = st.selectbox(
            "Select the target column", df.columns,
            index = default_index)

        target_col = st.session_state.target_col
        st.write("Selected target column: ", target_col)


        st.divider()
        st.subheader("Problem Type")

        options = ["Regression", "Classification"]

        st.session_state.problem_model = st.radio(
            "Select the problem",
            options,
            index=options.index(st.session_state.problem_model)
        )
        st.write("Problem type:", st.session_state.problem_model)
        

        
        st.divider()
        st.subheader("Target Distribution")

        y = df[target_col]

        col1, col2 = st.columns([1, 1])  

        with col1:
            if st.session_state.problem_model == "Classification":
                freq_df = y.value_counts().reset_index()
                freq_df.columns = ["Class", "Frequency"]

                fig, ax = plt.subplots(figsize=(4, 4))
                ax.bar(freq_df["Class"].astype(str), freq_df["Frequency"])
                ax.set_xlabel("Class")
                ax.set_ylabel("Count")
                ax.set_title("Class Distribution")
                st.pyplot(fig)

            elif st.session_state.problem_model == "Regression":
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(y.dropna(), bins=30)
                ax.set_xlabel("Target Value")
                ax.set_ylabel("Frequency")
                ax.set_title("Target Histogram")
                st.pyplot(fig)



elif page == "validation":
    st.title("Validation Page")

    if df is None:
        st.warning("First, load the dataset from the Data page.")
    else:
    
        if "test_size" not in st.session_state:
            st.session_state.test_size = 0.25

        st.session_state.test_size = st.slider(
            "Select the test size",
            0.05, 0.95,
            st.session_state.test_size,
            step=0.05
        )

        st.write("Test size:", st.session_state.test_size)

        



elif page == "preprocessing":
    st.title("Preprocessing Page")

    if df is None:
        st.warning("First, load the dataset from the Data page.")

    else:
        st.header("Feature Overview")
        st.write("Total number of columns: ", df.shape[1])

        categorical_columns = df.select_dtypes(include=["object", "string", "category"]).columns
        categorical_count = len(categorical_columns)
        st.write("Number of categorical columns: ", categorical_count)

        numeric_column = df.select_dtypes(include="number").columns
        numeric_count = len(numeric_column)
        st.write("Number of numeric columns: ",numeric_count)

        st.divider()
        st.subheader("Missing Value Percentage")
        st.write("Missing Value Percentage by Column")
        missing_df = pd.DataFrame({
            "column": df.columns,
            "missing_percent": [
                df[col].isna().sum() / df.shape[0] * 100
                for col in df.columns
            ]
        })

        missing_df = missing_df.sort_values(
            by="missing_percent",
            ascending=False
        )

        missing_df["missing_percent"] = missing_df["missing_percent"].round(2)

        st.write(missing_df)


        st.divider()
        st.subheader("Feature Drop")
        if "dropped_features" not in st.session_state:
            st.session_state.dropped_features = []

        st.session_state.dropped_features = st.multiselect(
            "Select the columns you want to drop",
            options=df.columns.tolist(),
            default=st.session_state.dropped_features
        )

        st.write("Columns to be dropped:", st.session_state.dropped_features)



elif page == "modeling":
    st.title("Modeling Page")
    if df is None:
        st.warning("First, load the dataset from the Data page.")

    else:
        st.subheader("Model Selection")

        if st.session_state.problem_model == "Classification":
            models = ["KNN", "Logistic Regression", "Random Forest", "Gradient Boosting"]
        else:  # Regression
            models = [
                "Multiple Linear Regression",
                "Random Forest Regressor",
                "Gradient Boosting Regressor",
                "XGBoost Regressor"
            ]


        if "selected_model" not in st.session_state:
            st.session_state.selected_model = models[0]

        if st.session_state.selected_model not in models:
            st.session_state.selected_model = models[0]

        st.session_state.selected_model = st.radio(
            "Select a model",
            models,
            index=models.index(st.session_state.selected_model),
            horizontal=True
        )

        st.write("Selected model:", st.session_state.selected_model)



        st.divider()
        st.subheader("Training Strategy")

        if "training_strategy" not in st.session_state:
            st.session_state.training_strategy = "Default Parameters"

        if (
            st.session_state.problem_model == "Regression"
            and st.session_state.selected_model == "Multiple Linear Regression"
        ):
            st.session_state.training_strategy = st.radio(
                "Parameter Strategy",
                ["Default Parameters"],
                index=0,
                horizontal=True
            )

        else:
            st.session_state.training_strategy = st.radio(
                "Parameter Strategy",
                ["Default Parameters", "Hyperparameter Tuning"],
                index=["Default Parameters", "Hyperparameter Tuning"].index(
                    st.session_state.training_strategy
                ),
                horizontal=True
            )

        run_config = {
            "problem": {
                "target_col": st.session_state.target_col,
                "problem_type": st.session_state.problem_model.lower()
            },
            "split": {
                "test_size": st.session_state.test_size,
                "random_state": 42
            },
            "feature_drop": {
                "columns": st.session_state.dropped_features
            },
            "model": {
                "name": st.session_state.selected_model
            },
            "training": {
                "strategy": (
                    "default"
                    if st.session_state.training_strategy == "Default Parameters"
                    else "tuning"
                )
            }
        }


        st.write("Training strategy:", st.session_state.training_strategy)


        st.divider()
        if st.button("Create Model", use_container_width=True):
            result = run_training_pipeline(df, run_config)
            st.write(result)
            


elif page == "evaluation":
    st.title("Evaluation Page")


    if "pipeline" not in st.session_state:
        st.warning("Train the model first.")

    else:
        metrics = evaluate_model(
            fitted_pipeline=st.session_state["pipeline"],
            X_test=st.session_state["X_test"],
            y_test=st.session_state["y_test"],
            problem_type=(
                "classification"
                    if st.session_state.problem_model == "Classification"
                    else "regression"
            )
        )

        st.subheader("Model Metrics")
        st.write(metrics)

        if st.session_state.problem_model == "Regression":

            y_true = st.session_state["y_test"]
            X_test = st.session_state["X_test"]
            pipeline = st.session_state["pipeline"]

            y_pred = pipeline.predict(X_test)
            residuals = y_true - y_pred

            col1, col2 = st.columns(2)

            with col1:
                fig1, ax1 = plt.subplots()
                ax1.scatter(y_true, y_pred, alpha=0.6)
                ax1.plot(
                    [y_true.min(), y_true.max()],
                    [y_true.min(), y_true.max()],
                    linestyle="--"
                )
                ax1.set_xlabel("Actual")
                ax1.set_ylabel("Predicted")
                ax1.set_title("Actual vs Predicted")
                st.pyplot(fig1)

            with col2:
                fig2, ax2 = plt.subplots()
                ax2.scatter(y_pred, residuals, alpha=0.6)
                ax2.axhline(0, linestyle="--")
                ax2.set_xlabel("Predicted")
                ax2.set_ylabel("Residuals")
                ax2.set_title("Residuals vs Predicted")
                st.pyplot(fig2)

            st.stop()


        st.divider()
        st.subheader("Evaluation Plots")

        y_true = st.session_state["y_test"]
        X_test = st.session_state["X_test"]
        pipeline = st.session_state["pipeline"]
        y_pred = pipeline.predict(X_test)

        col1, col2 = st.columns(2)

        with col1:
            cm = confusion_matrix(y_true, y_pred)

            fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax_cm, cmap=plt.cm.Blues, colorbar=False)
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)


        if (
            st.session_state.problem_model.lower() == "classification"
            and hasattr(pipeline.named_steps["model"], "predict_proba")
        ):
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            with col2:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)

                fig_roc, ax_roc = plt.subplots(figsize=(4, 4))
                ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax_roc.plot([0, 1], [0, 1], linestyle="--")
                ax_roc.set_title("ROC Curve")
                ax_roc.set_xlabel("FPR")
                ax_roc.set_ylabel("TPR")
                ax_roc.legend()
                st.pyplot(fig_roc)

            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = auc(recall, precision)

            col3, col4 = st.columns(2)
            with col3:
                fig_pr, ax_pr = plt.subplots(figsize=(4, 4))
                ax_pr.plot(recall, precision, label=f"AUC = {pr_auc:.2f}")
                ax_pr.set_title("Precision–Recall Curve")
                ax_pr.set_xlabel("Recall")
                ax_pr.set_ylabel("Precision")
                ax_pr.legend()
                st.pyplot(fig_pr)


elif page == "prediction":
    st.title("Prediction Page")

    if "pipeline" not in st.session_state:
        st.warning("Train the model first.")
    else:
        pipeline = st.session_state["pipeline"]

        st.subheader("Input Features")

        feature_names = (
            pipeline
            .named_steps["preprocess"]
            .feature_names_in_
        )

        input_data = {}

        for col in feature_names:
            if col in st.session_state.df.select_dtypes(include="number").columns:
                input_data[col] = st.number_input(
                    f"{col}",
                    value=0.0
                )
            else: #categorical
                unique_vals = (
                    st.session_state.df[col]
                    .dropna()
                    .unique()
                    .tolist()
                )

                input_data[col] = st.selectbox(
                    f"{col}",
                    options=unique_vals
                )

        input_df = pd.DataFrame([input_data])

        st.divider()

        if st.button("Make Prediction", use_container_width=True):

            prediction = pipeline.predict(input_df)[0]
            
            label_encoder = st.session_state.get("label_encoder")

            if label_encoder is not None:
                prediction = label_encoder.inverse_transform([prediction])[0]


            st.subheader("Prediction Result")
            st.success(f"Predict: {prediction}")

            if (
                st.session_state.problem_model.lower() == "classification"
                and hasattr(pipeline.named_steps["model"], "predict_proba")
            ):
                proba = pipeline.predict_proba(input_df)[0][1]
                st.info(f"Probability of being in the positive class: {proba:.2f}")

else:
    st.header("Welcome to the Homepage")
    st.write("Please go to the Data page to upload the dataset.")
