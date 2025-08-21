import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def visualize_class_samples(dataset_path, dataset_type="Original", samples_per_class=4):
    pass
    
def compare_original_vs_segmented(dataset_path, class_name="Early", number_pairs=3):
    pass
    

def analyze_pixel_distributions(dataset_path, data_type="Original",
                               class_name="Early", number_pairs=20):
    pass
    
if __name__== "__main__":
    # data path
    dataset_path = r"C:\Users\fenzi\repo_git\Acute_Lymphoblastic_Leukemia-Deep_Learning\dataset"
    
    print("===== VISUALIZING CLASS SAMPLES =====")
    
    # visualize samples from Original dataset
    visualize_class_samples(dataset_path, "Original", samples_per_class=4)
    
    # visualize samples from Segmented dataset
    visualize_class_samples(dataset_path, "Segmented", samples_per_class=4)
    
    print("\n ===== COMPARING ORIGINAL VS SEGMENTED =====")
    
    # compare Original vs Segmented for each class
    for class_name in ['Benign', 'Early', 'Pre', 'Pro']:
        compare_original_vs_segmented(dataset_path, class_name, number_pairs=4)
    
    print("\n ===== ANALYZING PIXEL DISTRIBUTIONS =====")
    
    # analyze pixel distributions for each class
    for class_name in ['Benign', 'Early', 'Pre', 'Pro']:
        print(f"\n Analyzing {class_name} class...")
        stats = analyze_pixel_distributions(dataset_path, "Original", class_name)
        print(f"Mean brightness: {stats['mean_brightness']:.2f}") # it say if a class is lighter or darker
        print(f"Brightness std: {stats['std_brightness']:.2f}") # it say how much brightness varies
        print(f"Pixel mean: {stats['pixel_mean']:.2f}") # overall avarage image intensity
        print(f"Pixel std: {stats['pixel_std']:.2f}") # over all contrast (high std = more contrast)