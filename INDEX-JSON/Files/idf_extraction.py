import json

file = open('C:\\Users\\Jaideep\\Desktop\\work\\INDEX-JSON\\newIndex.json','r')
data = json.load(file)

file2 = open('C:\\Users\\Jaideep\\Desktop\\work\\INDEX-JSON\\Files\\word_count.json','r')
data2 = json.load(file2)

doc_count = []
idf = []
tf = []
tf_idf = []

#highest_doc_freq_word = []
for key1, value1 in data.items():
	idf.append(value1['idf'])
	doc_count.append(len(value1['data']))

	if len(value1['data']) == 308:
		print('number 1 : ', key1)
	if len(value1['data']) == 238:
		print('number 2 : ', key1)
	if len(value1['data']) == 222:
		print('number 3 : ', key1)
	if len(value1['data']) == 174:
		print('number 4 :', key1)
	if len(value1['data']) == 151:
		print('number 5 :', key1)
	if len(value1['data']) == 143:
		print('number 6 :', key1)

	for items in value1['data']:
		for key2, value2 in data2.items():
			if items['doc-id'] == key2:
				ter_freq = items['tf']/value2
				tf.append(ter_freq)
				tf_idf_val = ter_freq * value1['idf']
				tf_idf.append(tf_idf_val)

with open('C:\\Users\\Jaideep\\Desktop\\work\\INDEX-JSON\\Files\\doc_count.txt', 'w') as out:
	for values in doc_count:
		out.write(str(values)+'\n')

with open('C:\\Users\\Jaideep\\Desktop\\work\\INDEX-JSON\\Files\\idf.txt', 'w') as out:
	for values in idf:
		out.write(str(values)+'\n')

with open('C:\\Users\\Jaideep\\Desktop\\work\\INDEX-JSON\\Files\\term_freq.txt', 'w') as out:
	for values in tf:
		out.write(str(values)+'\n')

with open('C:\\Users\\Jaideep\\Desktop\\work\\INDEX-JSON\\Files\\tf_idf.txt', 'w') as out:
	for values in tf_idf:
		out.write(str(values)+'\n')

#print(highest_doc_freq_word)