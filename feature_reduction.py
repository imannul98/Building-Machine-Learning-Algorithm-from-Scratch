import math
import sys
from typing import List
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class FeatureReduction(object):

    def __init__(self):
        pass

    @staticmethod
    def forward_selection(data: pd.DataFrame, target: pd.Series,
        significance_levels: List[float]=[0.01, 0.1, 0.2]) ->dict:
        """		
		Args:
		    data: (pandas data frame) contains the feature matrix
		    target: (pandas series) represents target feature to search to generate significant features
		    significance_levels: (list) thresholds to reject the null hypothesis
		Return:
		    significance_level_feature_map: (python map) contains significant features for each significance_level.
		    The key will be the significance level, for example, 0.01, 0.1, or 0.2. The values associated with the keys would be
		    equal features that has p-values less than the significance level.
		"""
        initial_features = data.columns.tolist()
        best_features = []
        significance_level_feature_map = {level: [] for level in significance_levels}
    
        while len(initial_features) > 0:
            remaining_features = list(set(initial_features) - set(best_features))
            new_pval = pd.Series(index=remaining_features)
        
            for new_column in remaining_features:
                model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
                new_pval[new_column] = model.pvalues[new_column]
        
            min_p_value = new_pval.min()
            if min_p_value < min(significance_levels):
                best_features.append(new_pval.idxmin())
            
                for level in significance_levels:
                    if min_p_value < level:
                        significance_level_feature_map[level] = best_features.copy()
            else:
                break
        return significance_level_feature_map
    
    @staticmethod
    def backward_elimination(data: pd.DataFrame, target: pd.Series,
        significance_levels: List[float]=[0.01, 0.1, 0.2]) ->dict:
        """		
		Args:
		    data: (pandas data frame) contains the feature matrix
		    target: (pandas series) represents target feature to search to generate significant features
		    significance_levels: (list) thresholds to reject the null hypothesis
		Return:
		    significance_level_feature_map: (python map) contains significant features for each significance_level.
		    The key will be the significance level, for example, 0.01, 0.1, or 0.2. The values associated with the keys would be
		    equal features that has p-values less than the significance level.
		"""
        features = data.columns.tolist()
        significance_level_feature_map = {level: [] for level in significance_levels}
        while len(features) > 0:
            features_with_constant = sm.add_constant(data[features])
            model = sm.OLS(target, features_with_constant).fit()
            p_values = model.pvalues[1:] 
            max_p_value = p_values.max()
        
            if max_p_value > max(significance_levels):
                features.remove(p_values.idxmax())
            else:
                for level in significance_levels:
                    significant_features = [feature for feature, p_value in p_values.items() if p_value < level]
                    if significant_features:
                        significance_level_feature_map[level] = significant_features
                break
        return significance_level_feature_map

    def evaluate_features(data: pd.DataFrame, y: pd.Series,
        significance_level_feature_map: dict) ->None:
        """
        PROVIDED TO STUDENTS

        Performs linear regression on the dataset only using the features discovered by feature reduction for each significance level.

        Args:
            data: (pandas data frame) contains the feature matrix
            y: (pandas series) output labels
            significance_level_feature_map: (python map) contains significant features for each significance_level. Each feature name is a string
        """
        min_rmse = sys.maxsize
        min_significance_level = 0
        for significance_level, features in significance_level_feature_map.items(
            ):
            removed_features = set(data.columns.tolist()) - set(features)
            print(
                f'significance level: {significance_level}, Removed features: {removed_features}'
                )
            data_curr_features = data[features]
            x_train, x_test, y_train, y_test = train_test_split(
                data_curr_features, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = math.sqrt(mse)
            print(f'significance level: {significance_level}, RMSE: {rmse}')
            if min_rmse > rmse:
                min_rmse = rmse
                min_significance_level = significance_level
        print(
            f'Best significance level: {min_significance_level}, RMSE: {min_rmse}'
            )
        print('')
