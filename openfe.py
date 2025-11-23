# import sys
# sys.path.append('../')
import pandas as pd
# from sklearn.datasets import fetch_california_housing
from openfe import OpenFE, tree_to_formula, transform
from sklearn.model_selection import train_test_split
import lightgbm as lgbpip
from sklearn.metrics import mean_squared_error
import itertools

def get_score(train_x, test_x, train_y, test_y):
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'seed': 1}
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = pd.DataFrame(gbm.predict(test_x), index=test_x.index)
    score = mean_squared_error(test_y, pred)
    return score

feature_categories = {
    'area': ['CA', 'PLAND', 'LPI'],
    'shape': ['TE', 'ED', 'LSI', 'SHAPE_MN', 'SHAPE_AM', 'PARA_MN', 'PARA_AM', 'PAFRAC'],
    'posui': ['PD', 'SPLIT', 'MESH'],
    'liantong': ['COHESION', 'AI', 'CLUMPY']
}
#CI、CD

if __name__ == '__main__':
    # 保存结果
    feature_combinations = {}
    n_jobs = 4
    origin = pd.read_excel('fin_data.xlsx')
    origin_aug = pd.concat([origin.copy() for _ in range(50)], axis=0, ignore_index=True)
    label = origin_aug[['CI_sum']]
    output_dir = './new_data'
    # 枚举组合数量从2到4（组合而非排列）
    for r in range(1, len(feature_categories) + 1):
        for combo in itertools.combinations(feature_categories.keys(), r):
            name = "_".join(combo)
            features = []
            for cat in combo:
                features.extend(feature_categories[cat])
            feature_combinations[name] = features
            data = origin_aug[features]
            data_origin = origin[features]
            # train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=1)
            # # get baseline score
            # score = get_score(train_x, test_x, train_y, test_y)
            # print("The MSE before feature generation is", score)
            # feature generation
            ofe = OpenFE()
            ofe.fit(data=data, label=label, n_jobs=n_jobs)
            feature_exprs = [tree_to_formula(f) for f in ofe.new_features_list[:32]]
            train_x ,train_x= transform(data_origin, data_origin ,ofe.new_features_list[:32], n_jobs=4)
            print(len(feature_exprs))
            train_x_new = train_x.iloc[:, -len(feature_exprs):]
            train_x_new.columns = feature_exprs
            train_x_new.to_csv(f"{output_dir}/{name}.csv", index=False)
            # OpenFE recommends a list of new features. We include the top 10
            # generated features to see how they influence the model performance
            # train_x, test_x = transform(train_x, test_x, ofe.new_features_list[:10], n_jobs=n_jobs)
            # score = get_score(train_x, test_x, train_y, test_y)
            # print("The MSE after feature generation is", score)
            # print("The top 10 generated features are")
            # for feature in ofe.new_features_list[:10]:
            #     print(tree_to_formula(feature))
