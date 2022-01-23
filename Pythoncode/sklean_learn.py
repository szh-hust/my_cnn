from sklearn import datasets


digits = datasets.load_digits()
print(digits.data[0])
print(digits.target)

"""
type(digits) <class 'sklearn.utils.Bunch'>
digits.keys()= dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])
images 是一个三维矩阵 1797 * 8 * 8
data 是具体数据，将8*8 的images按行展开成一行，共有1797 * 64
target 是一个1797维度的Vector，指明每张图片的标签，也就是每张图片代表的数字
target_names数据集中所有标签值，[0,1,2,3,4,5,6,7,8,9]
DESCR是一些作者的描述信息
"""

# print('digits.keys()=', digits.keys())
# digits.keys()= dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])

# print('digits.images.shape = ', digits.images.shape)
# digits.images.shape =  (1797, 8, 8)
# print('digits.images = ', digits.images)

# print('digits.data.shape = ', digits.data.shape)
# digits.data.shape =  (1797, 64)
# print('digits.data = ', digits.data)

# print('digits.target.shape = ', digits.target.shape)
# digits.target.shape =  (1797,)
# print('digits.target = ', digits.target)

# print('digits.target_names.shape = ', digits.target_names.shape)
# digits.target_names.shape =  (10,)
# print('digits.target_names = ', digits.target_names)
# digits.target_names =  [0 1 2 3 4 5 6 7 8 9]

# print(digits.DESCR)
