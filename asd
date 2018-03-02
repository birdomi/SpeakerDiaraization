summary ="""
{}.{}.{}
        learning_rate : {} train_data_ratio : {}  num_epoch : {}  batch_size : {}   windowsize : {} windowshift : {}		
        Best evaulatio based on test_data  :  Accuracy_train  : {}    Accuracy_test :  {}  best Result Matrix : \n{}\nat epoch :{}\n\n
        """.format(
    	rnn.name,experiment,msg,
    	rnn.learning_rate, train_rate, num_epoch,a.batch_size,a.windowsize,a.windowstep,			
    					stat["trains"],stat["tests"],stat["resultMatrix"],stat["epoch"])
