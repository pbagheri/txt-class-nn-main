# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:34:02 2019

@author: payam.bagheri
"""
# =============================================================================
# Libraries and General Functions
# =============================================================================
#import bigfoot_messaging_libraries_functions
runfile('C:/Users/Payam/Dropbox/Python/bigfoot_messaging_libraries_functions.py')

# =============================================================================
# Loading google's word2vec
# =============================================================================
# Load Google's pre-trained Word2Vec model.
# model_w2v = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/Payam/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

# =============================================================================
# Inputs
# =============================================================================
#print(dir_path)
mesdic = pd.read_excel(dir_path + '/0_input_data/bigfoot_messaging_dic.xlsx')
mesdic.columns
mesdic['statement'][0].split()
mesdic['statement'] = mesdic['statement'].str.lower()

unique_words_freq_dic_pickle = dir_path + '/0_input_data/unique_words_freq_dic.pickle'
pkl_file = open(unique_words_freq_dic_pickle, 'rb')
unique_words_freq_dic = pickle.load(pkl_file)
pkl_file.close()

tfidf_dict_pickle = dir_path + '/0_input_data/tfidf_dict.pickle'
pkl_file = open(tfidf_dict_pickle, 'rb')
tfidf_dict = pickle.load(pkl_file)
pkl_file.close()

#all_data = pd.read_excel(dir_path + '/0_input_data/messaging-open-end-all-quarters-english-cleaned.xlsx')
#all_data = shuffle(all_data, random_state=0)
#data = all_data

# This code prepares a dense version of the data were words that are similar enough to each other are turned into the same word 
runfile(dir_path + '/Python/bigfoot_messaging_spell_correction.py')

data = pd.read_csv(dir_path + '/0_input_data/bigfoot/bigfoot_message_dense_data.csv')


# =============================================================================
# Loading the dataset C (split-messsage data made by Andrew)
# ============================================================================= 
data_C = pd.read_excel(dir_path + '/0_input_data/dataset C - just sectoin data.xlsx')
data_C.columns

wordvec_size = 300

mess_vecs = np.zeros((data_C.shape[0],wordvec_size+1))

for i in tqdm(data_C.index):
    mess = data_C.aw_unaided_ad_message.loc[i]
    if not isNaN(mess):
        #print(mess)
        messpreped = mess_prep(mess)
        for word in messpreped:
            vec = get_vec(str(word))
            try:
                mess_vecs[i][0:wordvec_size] += tfidf_dict[word]*vec
            except KeyError:
                mess_vecs[i][0:wordvec_size] += vec
                
        mess_vecs[i][-1] = data_C.aw_unaided_ad_message_en_sroec1.loc[i]
    mess_vecs[i][0:wordvec_size] = mess_vecs[i][0:wordvec_size]/len(messpreped)

mess_vecs = pd.DataFrame(mess_vecs)



mess_vecs.dropna(inplace=True)
#mess_vecs = mess_vecs[mess_vecs['301'] != 0]
mess_vecs = shuffle(mess_vecs, random_state=0)

mess_vecs.columns

# =============================================================================
# Loading the dataset A (made by Andrew)
# ============================================================================= 
'''
data_A = pd.read_excel(dir_path + '/0_input_data/dataset A - just 500 rows.xlsx')
data_A.columns

wordvec_size = 300

mess_vecs = np.zeros((data_A.shape[0],wordvec_size+1))

for i in tqdm(data_A.index):
    mess = data_A.aw_unaided_ad_message.loc[i]
    if not isNaN(mess):
        #print(mess)
        messpreped = mess_prep(mess)
        for word in messpreped:
            vec = get_vec(str(word))
            try:
                mess_vecs[i][0:wordvec_size] += tfidf_dict[word]*vec
            except KeyError:
                mess_vecs[i][0:wordvec_size] += vec
                
        mess_vecs[i][-1] = data_A.aw_unaided_ad_message_en_sroec1.loc[i]
    mess_vecs[i][0:wordvec_size] = mess_vecs[i][0:wordvec_size]/len(messpreped)

mess_vecs = pd.DataFrame(mess_vecs)

mess_vecs.to_csv(dir_path + '/0_input_data/bigfoot/mess_vecs_dataset_C.csv', index = False)
'''

mess_vecs_dataset_C = pd.read_csv(dir_path + '/0_input_data/bigfoot/mess_vecs_dataset_C.csv')
mess_vecs_dataset_C.dropna(inplace=True)
#mess_vecs = mess_vecs[mess_vecs['301'] != 0]
mess_vecs_dataset_C = shuffle(mess_vecs_dataset_C, random_state=0)

mess_vecs_dataset_C.columns
mess_vecs_dataset_C.shape

mess_vecs_dataset_C['311'] = 2

# =============================================================================
# Data processing
# =============================================================================
#Run this if you want to use the tfidf values calculated within "class" as weights for word vectors
runfile(dir_path + '/Python/bigfoot_messaging_class_tfidf.py')
data_w2v_tfidf = pd.read_csv(dir_path + '/0_input_data/bigfoot/bigfoot_class_tfidf_vectors.csv')        


#Run this if you want to use the tfidf values calculated within "class" as weights for word vectors
runfile(dir_path + '/Python/bigfoot_messaging_class_tfidf_bagofwords.py')



#Run this if you want to use the tfidf values calculated within "message" and use the resulting tfidf-hot message vectors
runfile(dir_path + '/Python/bigfoot_messaging_message_tfidf.py')



# =============================================================================
# Loading message vectors 
# ============================================================================= 
# if you want the word2vec-vector-supplied class-tfidf-weighted message vectors
mess_vecs = pd.read_csv(dir_path + '/0_input_data/bigfoot/bigfoot_class_tfidf_vectors.csv')
mess_vecs.shape

# if you want the word2vec-vector-supplied class-tfidf-weighted message bagofwords vectors
mess_vecs_unstacked = pd.read_csv(dir_path + '/0_input_data/bigfoot/bigfoot_class_tfidf_bagofwords_vectors.csv', encoding="ISO-8859-1")
mess_vecs_unstacked.shape
mess_vecs_unstacked.columns
#mess_vecs = mess_vecs_unstacked.copy()


mess_vecs_unstacked.dropna(inplace=True)
mess_vecs_unstacked = shuffle(mess_vecs_unstacked, random_state=0)
mess_vecs = mess_vecs_unstacked.copy()

mess_vecs_unstacked.shape


mess_vecs_unstacked['311'] = 1



#=============================================================================
'''
mess_vecs_stacked = pd.DataFrame()

for i in range(300,310):
    col_list = [str(x) for x in range(300)]
    col_list.append(str(i))
    col_list.append('310')    
    temp_df = mess_vecs_unstacked[col_list]
    temp_df.rename(columns = {str(i): '300', '310':'301'}, inplace = True)
    mess_vecs_stacked = mess_vecs_stacked.append(temp_df)
    
mess_vecs_stacked.shape
mess_vecs_stacked.columns

mess_vecs_stacked.reset_index(inplace=True)
mess_vecs_stacked.dropna(inplace=True)
mess_vecs_stacked.shape
mess_vecs_stacked.columns
mess_vecs_stacked.drop(['index'], axis=1, inplace=True)
mess_vecs_stacked = mess_vecs_stacked[mess_vecs_stacked['300'] != 0]

mess_vecs_stacked.to_csv(dir_path + '/0_input_data/bigfoot/mess_vecs_stacked.csv', index = False)
'''

mess_vecs_stacked = pd.read_csv(dir_path + '/0_input_data/bigfoot/mess_vecs_stacked.csv', encoding="ISO-8859-1")
#mess_vecs = mess_vecs_stacked.copy()
mess_vecs_stacked = shuffle(mess_vecs_stacked, random_state=0)

# =============================================================================
# Concatenating Datasets
# =============================================================================
mess_vecs_unstacked_trimmed = mess_vecs_unstacked[list(mess_vecs_dataset_C.columns)]
mess_vecs_unstacked_trimmed.shape
mess_vecs_unstacked_trimmed.columns

mess_vecs = mess_vecs_unstacked_trimmed.append(mess_vecs_dataset_C)
mess_vecs = shuffle(mess_vecs, random_state=0)
mess_vecs.columns

#=============================================================================
mess_vecs.shape

#mess_vecs.to_csv(dir_path + '/0_input_data/bigfoot/test.csv', index = False)
#len(mess_vecs['300'].unique())

#mess_vecs[300]

#mess_vecs = pd.read_csv(dir_path + '/0_input_data/bigfoot/test.csv')

#valid_size = 0.1
train_percent = 0.8
valid_percent = 0.9

cols = mess_vecs.columns
num_train_set = int(train_percent*mess_vecs.shape[0])
num_valid_set = int(valid_percent*mess_vecs.shape[0])

train_vecs = mess_vecs[cols][0:num_train_set]
valid_vecs = mess_vecs[cols][num_train_set:num_valid_set]
#valid_vecs = valid_vecs[valid_vecs['311'] == 1]
test_vecs = mess_vecs[cols][num_valid_set:]
#stest_vecs = test_vecs[test_vecs['311'] == 1]

train_vecs.shape
valid_vecs.shape
test_vecs.shape

train_vecs.to_csv(dir_path + '/0_input_data/bigfoot_train_vectors.csv', index=False)
valid_vecs.to_csv(dir_path + '/0_input_data/bigfoot_valid_vectors.csv', index=False)
test_vecs.to_csv(dir_path + '/0_input_data/bigfoot_test_vectors.csv', index=False)


train_dataset = dir_path + '\\0_input_data\\bigfoot_train_vectors.csv'
valid_dataset = dir_path + '\\0_input_data\\bigfoot_valid_vectors.csv'
test_dataset = dir_path + '\\0_input_data\\bigfoot_test_vectors.csv'

# the -2 is for putting aside the last columns that are the labels and the message (in words) itself
 
num_words = 1
mess_length = num_words*300

# NETTING THE LEVELS
# the 1st list is the levels that are gonna keep their identity (although relabelled) and
# the 2nd list are the levels that are gonna be combined into one level
big_levels = [1, 8, 9]
mid_levels =[13, 5, 17, 16, 7, 2, 25, 4]
big_level_conv_dic = {1:1, 8:1, 9:1} 
mid_level_conv_dic = {13:2, 5:2, 17:2, 16:2, 7:2, 2:2, 25:2, 4:2}
to_be_combined_level = [14, 11, 18, 19, 3, 10, 12, 24, 6, 20, 15, 23, 26, 21, 22]

# Data loader machine
class DriveData(Dataset):
    def __init__(self, datasetf, transform=None):
        self.__xs = []
        self.__ys = []
        self.transform = transform
        # Open and load text file including the whole training data
        with open(datasetf) as f:
            #print(datasetf)
            i = 0
            for line in f:
                # the following i>0 is for skipping the first line which is the column names 
                if i > 0:
                    # checked the resizing to 50*300 and it's correct
                    #self.__xs.append(torch.from_numpy(np.asarray([float(x) for x in line.split(',')[0:15000]])).view(1,50,300))
                    self.__xs.append(torch.from_numpy(np.asarray([float(x) for x in line.split(',')[0:mess_length]])))
                    #if i == 5:
                        #print(i, self.__xs)
                    self.__ys.append(line.split(',')[300])
                i += 1
        f.close()

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        message = self.__xs[index]
        if self.transform is not None:
            message = self.transform(message)

        # Convert image and label to torch tensors
        #message = torch.from_numpy(np.asarray(message))
        # the subtraction of 1 is to make the target values range from 0 to 25 instead of 1 to 26
        #label = torch.from_numpy(np.asarray(int(float(self.__ys[index]))).reshape(1)) # when 0 is there
        #print(label)
        label = torch.from_numpy(np.asarray(int(float(self.__ys[index]))-1).reshape(1)) # when lables start from one
        #print(label)

# =============================================================================
#         if int(float(self.__ys[index])) in big_levels:
#             label = torch.from_numpy(np.asarray(big_level_conv_dic[int(float(self.__ys[index]))]-1).reshape(1))
#         elif int(float(self.__ys[index])) in mid_levels:
#             label = torch.from_numpy(np.asarray(mid_level_conv_dic[int(float(self.__ys[index]))]-1).reshape(1))
#         else:
#             label = torch.from_numpy(np.asarray(3-1).reshape(1))
# =============================================================================

        label = label.long()
        return message, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)
    
 
# number of subprocesses to use for data loading
num_workers = 0
batch_size = 20


# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([transforms.ToTensor()])

#indices = list(range(num_train))
#np.random.shuffle(indices)
#split = int(np.floor(valid_size * num_train))
#train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
#train_sampler = SubsetRandomSampler(train_idx)
#valid_sampler = SubsetRandomSampler(valid_idx)

train_data = DriveData(train_dataset, transform=None)
valid_data = DriveData(valid_dataset, transform=None)
test_data = DriveData(test_dataset, transform=None)

#train_loader = DataLoader(train_data, batch_size=1, num_workers=num_workers)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=1, num_workers=num_workers)

loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}


# =============================================================================
# Classification algorithm
# =============================================================================
# define the CNN architecture
# Each word is represented by a vector of length 300. The input matrix to the
# following matrix is a 40x300 matrix meaning each message can have up to a 
# length of 40 words. The three conv layers act on the input independently
# and then the 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(mess_length, int(mess_length/2))
        self.fc1 = nn.Linear(int(mess_length/2), int(mess_length/4))
        self.fc2 = nn.Linear(int(mess_length/4), int(mess_length/6))
        self.fc3 = nn.Linear(int(mess_length/6), 28)
        #self.fc4 = nn.Linear(1500, 500)
        #self.fc5 = nn.Linear(500, 100)
        #self.fc6 = nn.Linear(int(word_vec_size/12), 30)
        #self.sig = nn.Sigmoid()
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        #x = self.dropout(x)
        #x = F.relu(self.fc4(x))
        #x = self.dropout(x)
        #x = F.relu(self.fc5(x))
        #x = self.fc6(x)
        #x = self.sig(x)
        return x



# create a complete CNN
model = Net()
model = model.float() 
print(model)

model.apply(weights_init_normal)




### loss function
# It is useful when training a classification problem with C classes. 
# If provided, the optional argument weight should be a 1D Tensor assigning 
# weight to each of the classes. This is particularly useful when you have an 
# unbalanced training set.

# weight factors for the levels
#ratios = [0.02, 0.2360, 0.0425, 0.0130, 0.0370, 0.0603, 0.0045, 0.0464, 0.1332, 0.0827, 0.0100, 0.0272, 0.0084, 0.0749, 0.0282, 0.0010, 0.0535, 0.0561, 0.0207, 0.0172, 0.0013, 0.0006, 0.0000, 0.0010, 0.0052, 0.0382, 0.0010, 0, 0, 0]
#len(ratios)
#ratios = [0.235980551, 0.133225284, 0.082658023, 0.074878444, 0.060291734, 0.056077796, 0.053484603, 0.046353323, 0.042463533, 0.038249595, 0.036952998, 0.139384117]
#ratios_inv = [max(ratios)/(x+0.0001) for x in ratios]
#weights = torch.Tensor(ratios)

#criterion = nn.CrossEntropyLoss(weight=weights)

criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()

### optimizer
lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=lr)     

def train(n_epochs, loaders, model, optimizer, new_lr, criterion, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    valid_loss_prev = 0
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0         
        lower_lr = False
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            #print(batch_idx)
            if lower_lr == True:
                optimizer = optim.SGD(model.parameters(), lr=new_lr)
            ## find the loss and update the model parameters accordingly
            #print(new_lr)
            optimizer.zero_grad()
            output = model(data.float())
            #print(target.shape)
            target = target.squeeze_()
            #print(output.shape)
            
            loss = criterion(output, target)
            #print(loss)
        
            loss.backward()
        
            optimizer.step()
        
            #train_loss += loss.item()*data.size(0)
            ## record the average training loss, using something like
            train_loss += loss.data
        
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            ## update the average validation loss
            output = model(data.float())
            target = target.squeeze_()
            loss = criterion(output, target)
            valid_loss += loss.data
    
        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))

        ## save the model if validation loss has decreased
        if valid_loss_prev < valid_loss:
            #print('lr is being reduced at epoch %d' % epoch)
            new_lr = new_lr*0.95
            lower_lr = True
        valid_loss_prev = valid_loss
        
        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), save_path)
            model_min = model
            
    # return trained model
    return model_min


# train the model
#model_test = train(600, loaders, model, optimizer, lr, criterion, 'C:/Users/Payam/Dropbox/0_output/bigfoot_messaging_NN/model_messaging.pt')    

print('Training of the model starts!')
model_mess = train(200, loaders, model, optimizer, lr, criterion, 'C:/Users/Payam/Dropbox/0_output/bigfoot_messaging_NN/model_messaging.pt')

#model_mess = torch.load('C:/Users/Payam/Dropbox/0_output/bigfoot_messaging_NN/model_messaging.pt')

model_mess.eval()
test_results = []
preds = []
targs = []
test_loss = 0
for batch_idx, (data, target) in tqdm(enumerate(loaders['test'])):
    ## update the average validation loss
    output = model_mess(data.float())
    target = target.squeeze_()
    #loss = criterion(output, target)
    #print(output.shape,target.shape)
    pred = output.data.max(1, keepdim=True)[1]
    targs.append(int(target))
    preds.append(int(pred))
    #if int(target) == 0: 
        #print(int(pred),int(target))
    #test_loss += loss.data

present_codes = [0,	1,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,	14,	15,	16,	17,	18,	23,	24, 25]
names = ["hormones steroids",	"antibiotics",	"additives preservatives",	"local canadian",	"100% pure fresh real authentic",	"grass fed grafed",	"natural organic non-gmo",	"promotion coupon deal value price pricing cost quality",	"events shown ad manager interacting with people",	"root beer",	"great good",	"delicious great tasting",	"burgers patties",	"beef",	"chicken",	"dont know",	"irrelevent",	"bad respondent",	"quality premium",	"environmentally friendly sustainable farming",	"range",	"eggs",	"fish",	"vegetarian vegan  veggie meatless",	"beyond plant based pea protein",	"ingredients meat product food",	"tastes mimic similar",	"cheddar",]
len(names)
names_dict = dict(zip(range(28),names))
labels_list = [names_dict[x] for x in present_codes]

dict_names = dict(zip(names,range(28)))
print(classification_report(targs, preds))

print(classification_report(targs, preds,target_names =labels_list))
#print(test_loss/len(test_loader.dataset))

#test_loss = test_loss/len(test_loader.dataset)
#print(test_loss) 

# for multilabel **************************************************************
model_mess.eval()
test_results = pd.DataFrame(columns = range(10))
preds = []
targs = []
test_loss = 0
for batch_idx, (data, target) in tqdm(enumerate(loaders['test'])):
    ## update the average validation loss
    output = model_mess(data.float())
    target = target.squeeze_()
    #loss = criterion(output, target)
    #print(output.shape,target.shape)
    targs.append(int(target))
    output = softmax(output.tolist()[0])   
    indexed_output = [(i,x) for i, x in zip(range(1,28), output)]
    indexed_output = [x if x[1] > 0.3 else (0,x[1]) for x in indexed_output]
    indexed_output_sorted = sorted(indexed_output, key=takeSecond, reverse=True)
    probs = [x[1] for x in indexed_output_sorted]
    labels = [x[0] if x[0] > 0 else 0 for x in indexed_output_sorted]
    #print(labels)
    #lbs = pd.Series(labels[0:10])
    #test_results = test_results.append(lbs)
    test_results.loc[batch_idx] = labels[0:10]
    #test_loss += loss.data

targets = test_vecs[[str(x) for x in range(300,310)]]
targets.reset_index(inplace = True)
targets.drop(['index'], axis = 1, inplace = True)
#targets.rename(columns = dict(zip([str(x) for x in range(300,310)], range(0, 10))), inplace = True)
targets = targets.astype(int)

#targ_labl = targets.join(test_results)

targ_arr = np.array(targets)
labels = np.array(test_results)

mlb = MultiLabelBinarizer(classes = np.array(range(1,28)))
bin_targs = mlb.fit_transform(targ_arr)
bin_labs = mlb.fit_transform(labels)

print(classification_report(bin_targs, bin_labs))
    
test_results.to_csv(dir_path + '/0_output/test_results.csv', index = False)
targets.to_csv(dir_path + '/0_output/targets.csv', index = False)
targ_labl.to_csv(dir_path + '/0_output/targets_and_test_results.csv', index = False)


