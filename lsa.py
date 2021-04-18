import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix,vstack
from sklearn.cluster import KMeans
import numpy as np

# path='/mnt/data1/India/dataset/clusters_topic/cluster_0.csv'
path1='/mnt/data1/India/dataset/6_topic_germ_books.csv'
# df=pd.read_csv(path)
df = pd.read_csv(path1)
vectorizer = TfidfVectorizer(smooth_idf=True)
X = vectorizer.fit_transform(df['books'])
km = KMeans(n_clusters=7, max_iter=200, n_init=10, random_state=7)
km = km.fit(X)
terms = vectorizer.get_feature_names()
def ClusterIndicesNumpy(clustNum, labels_array): #numpy
    return np.where(labels_array == clustNum)[0]
for i in range(0,7):
    a = ClusterIndicesNumpy(i, km.labels_)
    tfidf = []
    for j in a:
        # book_name.append(df.book_name[j])
        tfidf.append(X[j].toarray())
# print(df.tfidf[0].toarray())
    tf = vstack(list(map(lambda y: csr_matrix(y), tfidf)))
    LSA_model = TruncatedSVD(n_components=1, algorithm='randomized', n_iter=100, random_state=7)
    LSA_model.fit(tf)
# len(LSA_model.components_)
    main_topic = []
    for k, comp in enumerate(LSA_model.components_):
        terms_comp = zip(terms, comp)
        l = []
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:20]
        # print("Topic "+str(k)+": ")
        for t in sorted_terms:
            print(t[0], end = " ")
            l.append(t[0])
        main_topic.append(",".join(l))
    print("topic"+str(i)+":")
    print(main_topic)
