array = np.array([])
        list1 = []
        list2 = []
for text in textList:
            text = text.split()
            #list1 = []
            for word in text:
                if word in game_vocab:
                    list1.append(game_vocab[word])
                else:
                    list1.append(int(word))
            #if len(list1) < 25:
                #for i in range((25-len(list1))):
                    #list1.insert(len(list1)+1,0)
            list2.append(sum(list1))
        #if len(list2)<5:
            #for i in range (5-len(list2)):
                #list2.append([0]*25)
        #print(sum(list1))
        print(list2[len(list2)-1])
        array = np.array(list1[len(list1)-1],dtype=np.int32)
        array = np.reshape(array,(1,1))
        #array = np.sum(array)
        #array = np.expand_dims(array, axis=0)
        #array = np.expand_dims(array, axis=0)
        print(array.shape)
        #tfarray = tf.convert_to_tensor(array)
        #array = np.reshape(array,(1,))
        #print(sumarray.shape,sumarray)
        print(array)
        return array