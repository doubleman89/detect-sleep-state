from matplotlib import pyplot as plt 

def plot_all (array):
    plt.figure(figsize=(15,5))
    array_len = int(array.shape[-1])
    for i in range(array_len):
        plt.subplot(array_len,1,i+1)
        plt.plot(array[:,i])