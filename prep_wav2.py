from scipy.io import wavfile
import argparse
import numpy as np
import random
import json

def save_wav(name, data):
    wavfile.write(name, 48000, data.flatten().astype(np.float32))

def save_wav_dont_flatten(name, data):
    wavfile.write(name, 48000, data.astype(np.float32))

def normalize(data):
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max,abs(data_min))
    if data_norm == 0:
        print("[WARNING]: Audio file appears to contain all 0's, indicating a completely silent wav file. Check your out.wav file.")
    return data / data_norm


def sliceOnMod(input_data, target_data, mod):
    # Split the data on a modulus.

    # type cast to an integer the modulus
    mod = int(mod)

    # Split the data into 100 pieces
    input_split = np.array_split(input_data, 100)
    target_split = np.array_split(target_data, 100)

    val_input_data = []
    val_target_data = []
    # Traverse the range of the indexes of the input signal reversed and pop every 5th for val
    for i in reversed(range(len(input_split))):
        if i%mod == 0:
            # Store the validation data
            val_input_data.append(input_split[i])
            val_target_data.append(target_split[i])
            # Remove the validation data from training
            input_split.pop(i)
            target_split.pop(i)

    # Flatten val_data down to one dimension and concatenate
    val_input_data = np.concatenate(val_input_data)
    val_target_data = np.concatenate(val_target_data)

    # Concatinate b back together
    training_input_data = np.concatenate(input_split)
    training_target_data = np.concatenate(target_split)
    return (training_input_data, training_target_data, val_input_data, val_target_data)

def sliceRandomPercentage(input_data, target_data, percentage):
    # Do a random split of the data by slicing the data into 100 pieces and randomly 
    # chosing some number of them.

    if percentage < 0 and percentage > 100:
        raise ValueError("Perentage must be between 0 and 100")

    # Split the data into 100 pieces
    input_split = np.array_split(input_data, 100)
    target_split = np.array_split(target_data, 100)
    validationChunks = int(percentage)

    # Skip the first entry because it has a different array size if the array can't be 
    # devided evenly which screws things
    selection = random.sample(range(1,99), validationChunks)

    val_input_data = []
    val_target_data = []
    # Store the randomly selected values in C
    for i, val in enumerate(selection):
        val_input_data.append(input_split[val])
        val_target_data.append(target_split[val])

    # Flatten val_data down to one dimension and concatenate
    val_input_data = np.concatenate(val_input_data)
    val_target_data = np.concatenate(val_target_data)

    # remove the validation selections from b
    for i in sorted(selection, reverse=True):
        input_split.pop(i)
        target_split.pop(i)

    # Concatinate b back together
    training_input_data = np.concatenate(input_split)
    training_target_data = np.concatenate(target_split)
    return (training_input_data, training_target_data, val_input_data, val_target_data)

def nonConditionedWavParse(args):
    # Load and Preprocess Data ###########################################
    in_rate, in_data = wavfile.read(args.snapshot[0])
    out_rate, out_data = wavfile.read(args.snapshot[1])

    clean_data = in_data.astype(np.float32).flatten()
    target_data = out_data.astype(np.float32).flatten()

    # If Desired Normalize the data
    if (args.normalize):
        clean_data = normalize(clean_data).reshape(len(clean_data),1)
        target_data = normalize(target_data).reshape(len(target_data),1)

    if args.random_split:
        in_train, out_train, in_val, out_val = sliceRandomPercentage(clean_data, target_data, args.random_split)
    else:
        # Split the data on a twenty percent mod
        in_train, out_train, in_val, out_val = sliceOnMod(clean_data, target_data, args.mod_split)

    # If separate test files are given process them.
    # Else, use validation data sets.
    if (len(args.snapshot) == 4):
        test_in_rate, test_in_data = wavfile.read(args.snapshot[2])
        test_out_rate, test_out_data = wavfile.read(args.snapshot[3])
        in_test = test_in_data.astype(np.float32).flatten()
        out_test = test_out_data.astype(np.float32).flatten()

        if (args.normalize):
            in_test = normalize(in_test).reshape(len(test_in_data),1)
            out_test = normalize(out_test).reshape(len(test_out_data),1)
    else:
        # It is NOT advisable to use the same data set for testing and validation
        # This gives a strong bias in evaluating the effectiveness of the training
        print("The test set and validation set are the same.")
        in_test = in_val
        out_test = out_val

    save_wav(args.path + "/train/" + args.name + "-input.wav", in_train)
    save_wav(args.path + "/train/" + args.name + "-target.wav", out_train)

    save_wav(args.path + "/test/" + args.name + "-input.wav", in_test)
    save_wav(args.path + "/test/" + args.name + "-target.wav", out_test)

    save_wav(args.path + "/val/" + args.name + "-input.wav", in_val)
    save_wav(args.path + "/val/" + args.name + "-target.wav", out_val)


def conditionedWavParse(args):
    '''
    This script processes multiple datasets to defined conditions of at
    most 65534 parameters due to the limited number of channels in a wav file.

    The script assumes that all parameters are normalized to a max of 1 (floating
    point 32).

    The processed "output" data will be a concatenated mono wav file.

    Note: Ensure to use the "is <num of parameters + 1>"  with dist_model_recnet.py
    on the conditioned data.

    Note: This is intended to be used with the colab conditioning training script.

    Note: Assumes all .wav files are mono, float32, no metadata
    '''
    # Open the configureation
    with open(args.parameterize, "r") as read_file:
        data = json.load(read_file)

    # Test to see if there are separate test data sets
    seprateTestSet = True
    try:
        a = data['Data Sets'][0]["TestClean"]
        a = data['Data Sets'][0]["TestTarget"]
    except KeyError:
        seprateTestSet = False
        print("The test set and validation set are the same.")


    params = data["Number of Parameters"]

    # Load and Preprocess Data ###########################################
    all_clean_train = np.array([[]]*(params+1)) # 1 channel for audio, n channels per parameters
    all_clean_val = np.array([[]]*(params+1)) # 1 channel for audio, n channels per parameters
    all_clean_test = np.array([[]]*(params+1)) # 1 channel for audio, n channels per parameters
    all_target_train = np.array([[]]) # 1 channels of all (out audio)
    all_target_val = np.array([[]]) # 1 channels of all (out audio)
    all_target_test = np.array([[]]) # 1 channels of all (out audio)


    for ds in data["Data Sets"]:

        # Load and Preprocess Data
        in_rate, in_data = wavfile.read(ds["TrainingClean"])
        out_rate, out_data = wavfile.read(ds["TrainingTarget"])
        
        if out_rate != 48000:
            print("\n\n\n[ERROR] The out*.wav file has an invalid samplerate (NeuralSeed / Daisy Seed uses 48kHz) " +"("+ str(out_rate) +")")
            print("[ERROR] Please re-export your wav file as 48000 samplerate (48kHz).\n\n\n")
            return
    
        if out_data.dtype != "int16" and out_data.dtype != "float32":
            print("\n\n\n[ERROR] The out*.wav file has an invalid data type " +"("+ str(out_data.dtype) +")")
            print("[ERROR] Please re-export your wav file as either PCM16 or FP32 data type (bit depth).\n\n\n")
            return        
        
        # Trim the length of audio to equal the smaller wav file
        if len(in_data) > len(out_data):
            print("Trimming input audio to match output audio")
            in_data = in_data[0:len(out_data)]
        if len(out_data) > len(in_data): 
            print("Trimming output audio to match input audio")
            out_data = out_data[0:len(in_data)]

        #If stereo data, use channel 0
        if len(in_data.shape) > 1:
            print("[WARNING] Stereo data detected for in_data, only using first channel (left channel)")
            in_data = in_data[:,0]
        if len(out_data.shape) > 1:
            print("[WARNING] Stereo data detected for out_data, only using first channel (left channel)")
            out_data = out_data[:,0]

        # Convert PCM16 to FP32
        if in_data.dtype == "int16":
            in_data = in_data/32767
            print("In data converted from PCM16 to FP32")
        if out_data.dtype == "int16":
            out_data = out_data/32767
            print("Out data converted from PCM16 to FP32")    
        
        clean_data = in_data.astype(np.float32).flatten()
        target_data = out_data.astype(np.float32).flatten()

        # If Desired Normalize the data
        if (args.normalize):
            clean_data = normalize(clean_data).reshape(len(clean_data),1)
            target_data = normalize(target_data).reshape(len(target_data),1)
            #in_test = normalize(in_test).reshape(len(test_in_data),1)
            #out_test = normalize(out_test).reshape(len(test_out_data),1)

        # Make the Training and validation split
        if args.random_split:
            in_train, out_train, in_val, out_val = sliceRandomPercentage(clean_data, target_data, args.random_split)
        else:
            # Split the data on a twenty percent mod
            in_train, out_train, in_val, out_val = sliceOnMod(clean_data, target_data, args.mod_split)


        # Process the Test Data. If it's a separate set, process that set. 
        # If there is not a separeate set, use the validation data for testing.
        if (seprateTestSet):
            test_in_rate, test_in_data = wavfile.read(ds["TestClean"])
            test_out_rate, test_out_data = wavfile.read(ds["TestTarget"])
            in_test = test_in_data.astype(np.float32).flatten()
            out_test = test_out_data.astype(np.float32).flatten()

            if (args.normalize):
                in_test = normalize(in_test).reshape(len(test_in_data),1)
                out_test = normalize(out_test).reshape(len(test_out_data),1)
        else:
            in_test = in_val
            out_test = out_val
            if (args.normalize):
                in_test = normalize(in_test).reshape(len(in_test),1)
                out_test = normalize(out_test).reshape(len(in_test),1)

        # Initialize lists to handle the number of parameters
        params_train = []
        params_val = []
        params_test = []

        # Create a list of np arrays of the parameter values
        for val in ds["Parameters"]:
            # Create the parameter arrays
            params_train.append(np.array([val]*len(in_train)))
            params_val.append(np.array([val]*len(in_val)))
            params_test.append(np.array([val]*len(in_test)))

        # Convert the lists to numpy arrays
        params_train = np.array(params_train)
        params_val = np.array(params_val)
        params_test = np.array(params_test)

        #print(np.array([in_train.flatten()]).shape)
        #print(params_train.shape)
        
        # Reformat for normalized data
        if (args.normalize):
            in_train = np.array([in_train.flatten()])
            in_val = np.array([in_val.flatten()])
            in_test = np.array([in_test.flatten()])
        
        # Append the audio and paramters to the full data sets 
        all_clean_train = np.append(all_clean_train, np.append(in_train,params_train, axis=0), axis = 1)
        all_clean_val = np.append(all_clean_val , np.append(in_val,params_val, axis=0), axis = 1)
        all_clean_test = np.append(all_clean_test , np.append(in_test,params_test, axis=0), axis = 1)

        all_target_train = np.append(all_target_train, out_train)
        all_target_val = np.append(all_target_val, out_val)
        all_target_test = np.append(all_target_test, out_test)

    # Save the wav files 
    save_wav_dont_flatten(args.path + "/train/" + args.name + "-input.wav", all_clean_train.T)
    save_wav_dont_flatten(args.path + "/val/" + args.name + "-input.wav", all_clean_val.T)
    save_wav_dont_flatten(args.path + "/test/" + args.name + "-input.wav", all_clean_test.T)

    save_wav(args.path + "/train/" + args.name + "-target.wav", all_target_train)
    save_wav(args.path + "/val/" + args.name + "-target.wav", all_target_val)
    save_wav(args.path + "/test/" + args.name + "-target.wav", all_target_test)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''This script prepairs the data data to be trained'''
    )
    parser.add_argument("name")
    # parser.add_argument("--in_file", "-in", type=str, default=None )
    # parser.add_argument("--out_file", "-out", type=str, default=None)
    # parser.add_argument("--test_in_file", "-tin", type=str, default=None)
    # parser.add_argument("--test_out_file", "-tout", type=str, default=None)
    parser.add_argument("--snapshot", "-s", nargs="+", help="Snapshot configuration. TRAINING_IN TRAINING_OUT OPTIONAL_TEST_IN OPTIONAL_TEST_OUT")
    parser.add_argument("--normalize", "-n", type=bool, default=False)
    parser.add_argument("--parameterize", "-p", type=str, default=None)
    parser.add_argument("--mod_split", '-ms', default=5, help="The default splitting mechanism. Splits the training and validation data on a 5 mod (or 20 percent).")
    parser.add_argument("--random_split", '-rs', type=float, default=None, help="By default, the training is split on a modulus. However, desingnating a percentage between 0 and 100 will create a random data split between the training and validatation sets.")
    parser.add_argument("--path", type=str, default="Data", help="Path to store the processed data.")

    args = parser.parse_args()

    if args.parameterize:
        print("Parameterized Data")
        conditionedWavParse(args)

    else:
        print("Non-Parameterized Data")
        nonConditionedWavParse(args)
