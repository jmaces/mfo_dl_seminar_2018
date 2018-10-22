import numpy as np
import os
import sys

# searches for results files in a directory and reduces data into a single file
def collect_results(directory):
    print('Searching for result files in {}.'.format(directory))
    try:
        filecount = 0
        depthlist = []
        widthlist = []
        # loop through all result files and retrieve parameter ranges
        for filename in os.listdir(directory):
            if filename.endswith(".npz"): 
                filecount +=1
                data = np.load(os.path.join(directory, filename))
                depthlist.append(int(data['depth']))
                widthlist.append(int(data['width'])) 
        print('Found {} result files.'.format(filecount))
        depthlist = sorted(set(depthlist))
        widthlist = sorted(set(widthlist))
        print('Found {} distinct depth parameters.'.format(len(depthlist)))
        print('Found {} distinct width parameters.'.format(len(widthlist)))

        # placeholders for results of all architectures from training
        RELU_DW     = np.nan*np.ones([len(depthlist), len(widthlist)])
        SIGMOID_DW  = np.nan*np.ones([len(depthlist), len(widthlist)])


        # loop through all result files again and retrieve data
        for filename in os.listdir(directory):
            if filename.endswith(".npz"): 
                data = np.load(os.path.join(directory, filename))

                depth_idx = depthlist.index(data['depth'])
                width_idx = widthlist.index(data['width'])

                if data['activation'] == 'relu':
                    RELU_DW[depth_idx, width_idx] = data['linf_error']
                    
                if data['activation'] == 'sigmoid':
                    SIGMOID_DW[depth_idx, width_idx] = data['linf_error']
        
        # save collected results
        np.savez_compressed(
            os.path.join(directory, 'collected_results.npz'),
            depthlist=depthlist,
            widthlist=widthlist,
            relu_error=RELU_DW,
            sigmoid_error=SIGMOID_DW
        )
        print('Saved results in {} as collected_results.npz'.format(directory))  
    except Exception as err:
        print('Something went wrong...')
        print(err)


# run collect results in a given directory
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide a directory name to search for result files...')
    else:
        directory = sys.argv[1]
        collect_results(directory)
