import os
fn = os.path.join(os.path.dirname(__file__), 'a3_v2_diag.py')
content = open(fn, 'r', encoding='utf-8').read()

fixes = 0
# 1) Replace hardcoded feature_names with dynamic feat_names
#    but only inside walk_forward_eval, not the final training at bottom
lines = content.split('\n')
new_lines = []
in_wfe = False
for i, line in enumerate(lines):
    if 'def walk_forward_eval' in line:
        in_wfe = True
    if in_wfe and line.strip().startswith('def ') and 'walk_forward_eval' not in line:
        in_wfe = False
    
    if in_wfe and 'feature_names=V1_FEATURES+MKT_FEATURES' in line:
        line = line.replace('feature_names=V1_FEATURES+MKT_FEATURES', 'feature_names=feat_names')
        fixes += 1
    new_lines.append(line)

content = '\n'.join(new_lines)

# 2) Insert n_feats/feat_names right before the first X_tr DMatrix creation in walk_forward_eval
target = '        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)'
insert = '''        n_feats = X_tr.shape[1]
        feat_names = V1_FEATURES[:n_feats] + (MKT_FEATURES if n_feats > len(V1_FEATURES) else [])
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)'''
content = content.replace(target, insert, 1)
fixes += 1

# 3) Same for the subsampled block if not already handled
target2 = '''        if len(X_tr) > 300000:
            idxs = np.random.RandomState(42).permutation(len(X_tr))[:300000]
            X_tr_sub = X_tr[idxs]; y_tr_sub = y_tr[idxs]
            dtrain = xgb.DMatrix(X_tr_sub, label=y_tr_sub, 
feature_names=feat_names)'''
insert2 = '''        if len(X_tr) > 300000:
            idxs = np.random.RandomState(42).permutation(len(X_tr))[:300000]
            X_tr_sub = X_tr[idxs]; y_tr_sub = y_tr[idxs]
            n_feats = X_tr_sub.shape[1]
            feat_names = V1_FEATURES[:n_feats] + (MKT_FEATURES if n_feats > len(V1_FEATURES) else [])
            dtrain = xgb.DMatrix(X_tr_sub, label=y_tr_sub, 
feature_names=feat_names)'''
content = content.replace(target2, insert2, 1)
fixes += 1

open(fn, 'w', encoding='utf-8').write(content)
print(f'Patched {fixes} locations OK')
