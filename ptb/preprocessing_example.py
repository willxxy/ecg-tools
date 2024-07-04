from collections import Counter
import argparse
import pickle

from ptb_utils import *

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seg_len', type = int, default = None, help='Please choose the segment length')
    return parser.parse_args()

def main(args):
    '''
        LABEL LEGEND FOR PTB
        {0: 'CD', 
        1: 'HYP', 
        2: 'MI', 
        3: 'NORM', 
        4: 'STTC'}    
        '''

    sampling_frequency=500
    datafolder='./ptb/'
    task='superdiagnostic'
    outputfolder='./new_ptb/'

    # Load PTB-XL data
    data, raw_labels = load_dataset(datafolder, sampling_frequency)
    # Preprocess label data
    labels = compute_label_aggregations(raw_labels, datafolder, task)
    # Select relevant data and convert to one-hot
    data, labels, Y, _ = select_data(data, labels, task, min_samples=0, outputfolder=outputfolder)

    # 1-7 for training 
    X_train = data[labels.strat_fold < 8]
    y_train = Y[labels.strat_fold < 8]
    y_train = np.argmax(y_train, axis=1)

    # 8 for validation
    X_val = data[labels.strat_fold == 8]
    y_val = Y[labels.strat_fold == 8]
    y_val = np.argmax(y_val, axis=1)

    # 9-10 for validation
    X_test = data[labels.strat_fold > 8]
    y_test = Y[labels.strat_fold > 8]
    y_test = np.argmax(y_test, axis=1)

    # print(y_train)
    # [2 3 3 ... 2 3 3]
    # print(X_train.shape, X_val.shape, X_test.shape)
    #(14955, 5000, 12) (2129, 5000, 12) (4304, 5000, 12)

    train_label_counts = Counter(y_train)
    val_label_counts = Counter(y_val)
    test_label_counts = Counter(y_test)

    # print(train_label_counts)
    # Counter({3: 6370, 0: 3415, 2: 2201, 4: 1662, 1: 1307})
    # print(val_label_counts)
    # Counter({3: 893, 0: 492, 2: 316, 4: 241, 1: 187})
    # print(test_label_counts)
    # Counter({3: 1834, 0: 991, 2: 614, 4: 497, 1: 368})
    if args.seg_len is not None:
        X_train, y_train = segment_ecg_data(X_train, y_train, args.seg_len)
        X_val, y_val = segment_ecg_data(X_val, y_val, args.seg_len)
        X_test, y_test = segment_ecg_data(X_test, y_test, args.seg_len)

        print(X_train.shape, X_val.shape, X_test.shape)
        # when args.seg_len = 1000 (74775, 1000, 12) (10645, 1000, 12) (21520, 1000, 12)
        # when args.seg_len = 500 (149550, 500, 12) (21290, 500, 12) (43040, 500, 12)
    
    assert len(y_train) == X_train.shape[0] and len(y_val) == X_val.shape[0] and len(y_test) == X_test.shape[0]

    standard_scaler = pickle.load(open('./standard_scaler.pkl', "rb"))

    X_train = apply_standardizer(X_train, standard_scaler)
    X_val = apply_standardizer(X_val, standard_scaler)
    X_test = apply_standardizer(X_test, standard_scaler)

    X_train = np.transpose(X_train, (1, 2, 0))
    X_val = np.transpose(X_val, (1, 2, 0))
    X_test = np.transpose(X_test, (1, 2, 0))

    # print(X_train.shape, X_val.shape, X_test.shape)
    # (1000, 12, 74775) (1000, 12, 10645) (1000, 12, 21520)

    if args.seg_len == None:
        args.seg_len = X_train.shape[0]
        
    train_dic = {}
    val_dic = {}
    test_dic = {}
    for i in range(len(y_train)):
        train_dic[(0, i, y_train[i])] = X_train[:, :, i].reshape(12, args.seg_len)
    for i in range(len(y_val)):
        val_dic[(0, i, y_val[i])] = X_val[:, :, i].reshape(12, args.seg_len)
    for i in range(len(y_test)):
        test_dic[(0, i, y_test[i])] = X_test[:, :, i].reshape(12, args.seg_len)

    print(len(train_dic))
    print(len(val_dic))
    print(len(test_dic))

    np.save(f'./train_data_ptb_{args.seg_len}.npy', train_dic)
    np.save(f'./val_data_ptb_{args.seg_len}.npy', val_dic)
    np.save(f'./test_data_ptb_{args.seg_len}.npy', test_dic)
        
    # inst = X_train[0, :, 0]
    # time = np.arange(args.seg_len)

    # # # Plotting the time series
    # plt.plot(time, inst)
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.title('Time Series Plot')
    # plt.savefig(f'./ecg_{args.seg_len}.png')
    # plt.close()
    # print('done')

if __name__ == '__main__':
    args = get_args()
    main(args)
