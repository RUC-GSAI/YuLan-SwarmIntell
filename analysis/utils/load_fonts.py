import os 
import matplotlib.pyplot as plt


try:
    font_path = r'./assets/fonts/Helvetica'
    if os.path.exists(font_path):
        from matplotlib.font_manager import fontManager
        for fname in os.listdir(font_path):
            if fname.lower().endswith('.ttf'):
                fontManager.addfont(path=os.path.join(font_path, fname))
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['Helvetica Now Text ', 'HelveticaNeue', 'Helvetica', 'Arial']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.linewidth'] = 1.2
        print("Attempting to use Helvetica font.")
    else:
        print(f"Helvetica font path '{font_path}' not found, using default sans-serif.")
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.linewidth'] = 1.2
except Exception as e:
    print(f"Error setting up Helvetica font: {e}. Using default sans-serif.")
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
