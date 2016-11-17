import rpc_classifier


T1, T1Labels = make_blobs(n_samples=30, centers=[[2 for x in range(2)],[0 for x in range(2)]], n_features=2)
T2, T2Labels = make_blobs(n_samples=10, centers=[[2 for x in range(2)],[0 for x in range(2)]], n_features=2)
clf = RPC_Classifier(proto_init_type='dataset', loglevel='debug', lmbd=1000, iter_num = 10)
clf.fit(T1=T1, T2=T2, T1Labels=T1Labels)
print clf.predict()
print clf.score(real_labels=T2Labels)
