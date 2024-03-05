import fasttext


print("\nFasttext : ")
model = fasttext.train_supervised(input='data.train')
print(model.test('data.val'))