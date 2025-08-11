import os
import traceback
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

def exploration_datset(dataset_path, dataset_type="Original"):
    """
    description: function for exploration dataset and know the statistics in it
    
    args: dataset_path (path data), dataset_type (typology of dataset)
    
    """
    print(f"\n Dataset analysis: {dataset_type.upper()}")
    
    base_path = os.path.join(dataset_path, dataset_type)
    classes = ['Benign', 'Early', 'Pre', 'Pro']
    
    stats = {
        'classes_counts': {},
        'image_sizes': [],
        'formats': [],
        'total_images': 0
    }
    
    # analysis for classes
    for class_name in classes:
        class_path = os.path.join(base_path, class_name)
        
        if os.path.exists(class_path):
            # count images
            images = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            stats['classes_counts'][class_name] = len(images)
            stats['total_images'] += len(images)
            
            print(f"{class_name}: {len(images)} images found")
            
            # analize for size and formats the first 50
            for i, img_file in enumerate(images[:50]):
                img_path = os.path.join(class_path, img_file)
                try:
                    with Image.open(img_path) as img:
                        stats['image_sizes'].append(img.size)
                        stats['formats'].append(img.format)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        else:
            stats['classes_counts'][class_name] = 0
            print(f"ERROR! - Directory {class_name} not found at {class_path}")    
    return stats

def visualize_dataset_stats(stats_original, stats_segmented):
    """
    description: visualization dataset statistics
    
    args: stats_original, stats_segmented (paths of sub-directory dataset)
    """
    fig, axes = plt.subplots(2,2, figsize=(15,10))
    fig.suptitle('Analysis of Acute Lymphoblastic Leukemia', 
                 fontsize=16, fontweight='bold')
    
    # graph 1 - class distribution (Original)
    classes = list(stats_original['classes_counts'].keys())
    count_original = list(stats_original['classes_counts'].values())
    
    axes[0,0].bar(classes, count_original, color='skyblue', alpha=0.7)
    axes[0,0].set_title('Class distribution - Original Dataset')
    axes[0,0].set_ylabel('Number Images')
    for i, v in enumerate(count_original):
        axes[0,0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        
    # graph 2 - class distribution (Segmented)
    count_segmented = list(stats_segmented['classes_counts'].values())
    
    axes[0,1].bar(classes, count_segmented, color='lightcoral', alpha=0.7)
    axes[0,1].set_title('Class distribution - Segmented Dataset')
    axes[0,1].set_ylabel('Number Images')
    for i, v in enumerate(count_segmented):
        axes[0,1].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

    # graph 3 - Original VS Segmented
    x = np.arange(len(classes))
    width = 0.35
    
    axes[1,0].bar(x - width/2, count_original, width, label='Original', color='skyblue', alpha=0.7)
    axes[1,0].bar(x + width/2, count_segmented, width, label='Segmented', color='lightcoral', alpha=0.7)
    axes[1,0].set_title('Comparison between Original and Segmented')
    axes[1,0].set_ylabel('Number Images')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(classes)
    axes[1,0].legend()
    
    # graph 4 - size images (both datasets)
    all_sizes_original = stats_original.get('image_sizes', [])
    all_sizes_segmented = stats_segmented.get('image_sizes', [])
    
    if all_sizes_original or all_sizes_segmented:
        # chart Original sizes
        if all_sizes_original:
            widths_orig = [s[0] for s in all_sizes_original]
            heights_orig = [s[1] for s in all_sizes_original]
            axes[1,1].scatter(widths_orig, heights_orig, alpha=0.6, 
                              color='skyblue', label='Original', s=20)
            
            # chart Segmented sizes
            if all_sizes_segmented:
                widths_seg = [s[0] for s in all_sizes_segmented]
                heights_seg = [s[1] for s in all_sizes_segmented]
                axes[1,1].scatter(widths_seg, heights_seg, alpha=0.6,
                                  color='lightcoral', label='Segmented', s=20)
                
            axes[1,1].set_title('image sizes distribution (sample)')
        axes[1,1].set_xlabel('width (px)')
        axes[1,1].set_ylabel('height (px)')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
    else:
        axes[1,1].text(0.5, 0,5, 'No size data available',
                       ha='center', va='center', transfrom=axes[1,1].transAxes)
        axes[1,1].set_title('Image Sizes - No Data')
    
    plt.tight_layout()
    plt.show()

def dataset_summary(stats_original, stats_segmented):
    """ 
    description: visualization of statistics
    
    args: stats_original, stats_segmented (stats dictionaries)
    """
    
    print("\n ===== SUMMARY DATASET =====")
    
    # general stats
    total_original = stats_original['total_images']
    total_segmented = stats_segmented['total_images']
    
    print(f"Total images Original: {total_original}")
    print(f"Total images Segmented: {total_segmented}")
    print(f"Total combined: {total_original + total_segmented}")
    
    # balance between classes
    print("\n Balance between classes:")
    for class_name in ['Benign', 'Early', 'Pre', 'Pro']:
        original_count = stats_original['classes_counts'].get(class_name, 0)
        segmented_count = stats_segmented['classes_counts'].get(class_name, 0)
        
        original_percentual = (original_count / total_original *100) if total_original > 0 else 0
        segmented_percentual = (segmented_count / total_segmented *100) if total_segmented > 0 else 0

        print(f"{class_name:8}: Original {original_count:4} ({original_percentual:5.1f}%) | "
              f"segmented {segmented_count:4} ({segmented_percentual:5.1f}%)")
        
    # images formats (Original)
    if stats_original['formats']:
        format_counts = Counter(stats_original['formats'])
        print("\n Formats images")
        for format, count in format_counts.items():
            print(f"{format}: {count} images")
            
    # images formats (Segmented)
    if stats_segmented['formats']:
        print("Segmented Dataset:")
    if stats_segmented['formats']:
        format_counts = Counter(stats_segmented['formats'])
        for format_type, count in format_counts.items():
            print(f"     {format_type}: {count} images")
    else:
        print("     No format data available")
        
    # Most common image sizes
    print(f"\n MOST COMMON IMAGE SIZES:")
    print("   Original Dataset:")
    if stats_original['image_sizes']:
        size_counts = Counter(stats_original['image_sizes'])
        for size, count in size_counts.most_common(5):
            print(f"{size[0]}x{size[1]}: {count} images")
    else:
        print("No size data available")
        
    print("Segmented Dataset:")
    if stats_segmented['image_sizes']:
        size_counts = Counter(stats_segmented['image_sizes'])
        for size, count in size_counts.most_common(5):
            print(f"     {size[0]}x{size[1]}: {count} images")
    else:
        print("No size data available")

def check_directory_structure(dataset_path):
    """
    Check if the directory structure is correct
    """
    print("\n ===== CHECKING DIRECTORY STRUCTURE =====")
    
    required_structure = {
        'Original': ['Benign', 'Early', 'Pre', 'Pro'],
        'Segmented': ['Benign', 'Early', 'Pre', 'Pro']
    }
    
    for main_dir, subdirs in required_structure.items():
        main_path = os.path.join(dataset_path, main_dir)
        print(f"\n {main_dir} directory:")
        
        if os.path.exists(main_path):
            print(f" {main_path} exists")
            
            for subdir in subdirs:
                subdir_path = os.path.join(main_path, subdir)
                if os.path.exists(subdir_path):
                    file_count = len([f for f in os.listdir(subdir_path) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
                    print(f"{subdir}: {file_count} files")
                else:
                    print(f"{subdir}: NOT FOUND")
        else:
            print(f"{main_path} NOT FOUND")
            
if __name__ == "__main__":
    # data path
    dataset_path = r"C:\Users\fenzi\repo_git\Acute_Lymphoblastic_Leukemia-Deep_Learning\dataset"
    
    # check directory structure
    check_directory_structure(dataset_path)
    
    print("\n ====== STARTING DATASET EXPLORATION ======")
    
    try:
        # exploration both datasets
        stats_original = exploration_datset(dataset_path, "Original")
        stats_segmented = exploration_datset(dataset_path, "Segmented")
        
        # visualization results
        dataset_summary(stats_original, stats_segmented)
        
        # create charts
        visualize_dataset_stats(stats_original, stats_segmented)
        
    except Exception as e:
        print(f"Error during exploration: {e}")
        traceback.print_exc()