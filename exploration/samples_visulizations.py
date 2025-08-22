import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def visualize_class_samples(dataset_path, dataset_type="Original", samples_per_class=4):
    """
    description:
        visualize samples from each class to understand the data
    
    args:
        dataset_path,
        dataset_type,
        samples_per_class
    """
    base_path = os.path.join(dataset_path, dataset_type)
    classes = ['Benign', 'Early', 'Pre', 'Pro']
    
    fig, axes = plt.subplots(len(classes), samples_per_class,
                             figsize=(16, 12))
    fig.suptitle(f'Leukemia Dataset Samples - {dataset_type}',
                 fontsize=16, fontweight='bold')
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(base_path, class_name)
        
        if os.path.exists(class_path):
            # get random samples
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            sample_images = random.sample(images,
                                          min(samples_per_class, len(images)))
            
            for img_idx, img_file in enumerate(sample_images):
                img_path = os.path.join(class_path, img_file)
                
                try:
                    img = Image.open(img_path)
                    
                    # display image
                    axes[class_idx, img_idx].imshow(img)
                    axes[class_idx, img_idx].axis('off')
                    
                    # add title only to first image of each row
                    if img_idx == 0:
                        axes[class_idx, img_idx].set_ylabel(
                            f'{class_name}\n(n={len(images)})',
                            fontweight='bold', fontsize=12, rotation=0,
                            ha='right', va='center'
                        )
                        
                    # add image filename as subtitle
                    axes[class_idx, img_idx].set_title(f'{img_file[:15]}...',
                                                       fontsize=8)
                    

                except Exception as e:
                    axes[class_idx, img_idx].text(0.5, 0.5, f'Error\n{str(e)[:20]}',
                                                  ha='center', va='center',
                                                  transform=axes[class_idx, img_idx].transAxes)
                    axes[class_idx, img_idx].axis('off')
    
    plt.tight_layout()
    plt.show()

def compare_original_vs_segmented(dataset_path, class_name="Early", number_pairs=3):
    """
    description:
        compare Original vs Segmented images to understand preprocessing
    
    args:
        dataset_path,
        class_name,
        number_pairs
    """
    original_path = os.path.join(dataset_path, "Original", class_name)
    segmented_path = os.path.join(dataset_path, "Segmented", class_name)
    
    fig, axes = plt.subplots(2, number_pairs, figsize=(15,8))
    fig.suptitle(f'Original vs Segmented Comparison - {class_name} Class',
                 fontsize=16, fontweight='bold')
    
    # get image lists
    original_images = [f for f in os.listdir(original_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    segmented_images = [f for f in os.listdir(segmented_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # sample random images
    sample_images = random.sample(original_images, min(number_pairs, len(original_images)))
    
    for i, img_name in enumerate(sample_images):
        # original image
        orig_img_path = os.path.join(original_path, img_name)
        try:
            orig_img = Image.open(orig_img_path)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original\n{img_name[:20]}...', fontsize=10)
            axes[0, i].axis('off')
        except Exception as e:
            axes[0, i].text(0.5, 0.5, f'Error: {str(e)[:20]}',
                            ha='center', va='center',
                            transform=axes[0, i].transAxes)
            axes[0, i].axis('off')
        
        # segmented image (try to find matching or use random)
        seg_img_path = os.path.join(segmented_path, img_name)
        if not os.path.exists(seg_img_path) and segmented_images:
            seg_img_path = os.path.join(segmented_path, random.choice(segmented_images))

        if os.path.exists(seg_img_path):
            try:
                seg_img = Image.open(seg_img_path)
                axes[1, i].imshow(seg_img)
                axes[1, i].set_title(f'Segmented\n{os.path.basename(seg_img_path)[:20]}...',
                                     fontsize=10)
                axes[1, i].axis('off')
            except Exception as e:
                axes[1, i].text(0.5, 0.5, f'Error: {str(e)[:20]}',
                                ha='center', va='center',
                                transform=axes[1, i].transAxes)
                axes[1, i].axis('off')

    # add row labels
    axes[0,0].set_ylabel('ORIGINAL', fontweight='bold', fontsize=14,
                         rotation=90, ha='right', va='center')
    axes[1,0].set_ylabel('SEGMENTED', fontweight='bold', fontsize=14,
                         rotation=90, ha='right', va='center')
    
    plt.tight_layout()
    plt.show()

def analyze_pixel_distributions(dataset_path, data_type="Original",
                               class_name="Early", number_pairs=20):
    pass
    
if __name__ == "__main__":
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