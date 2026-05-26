import os
import sys
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import dose_pre.myDoseNet as myDoseNet


def crop_from_pos(center_point_index, image, patch_size=(32, 32, 32)):
    # center_point = center_point
    center_x, center_y, center_z = map(int, center_point_index)
   
    half_size = [size // 2 for size in patch_size]
    start_x = center_x - half_size[0]
    start_y = center_y - half_size[1]
    start_z = center_z - half_size[2]
    
    size = patch_size
   
    image_size = image.GetSize()
   
    if start_x < 0:
        start_x = 0
    if start_x > image_size[0] - patch_size[0]:
        start_x = image_size[0] - patch_size[0]
        
    if start_y < 0:
        start_y = 0
    if start_y > image_size[1] - patch_size[1]:
        start_y = image_size[1] - patch_size[1]
        
    if start_z < 0:
        start_z = 0
    if start_z > image_size[2] - patch_size[2]:
        start_z = image_size[2] - patch_size[2]
   
    cropped_image = sitk.Extract(
        image,
        size=size,
        index=[start_x, start_y, start_z]
    )


    return cropped_image

def get_max_intensity_coordinate(image: sitk.Image) -> tuple:
   
    image_array = sitk.GetArrayFromImage(image)
    max_index = np.unravel_index(np.argmax(image_array, axis=None), image_array.shape)
    # max_coordinate = image.TransformIndexToPhysicalPoint(max_index[::-1])
    
    return max_index

def pad_to_original_size(cropped_image: sitk.Image, 
                        original_image: sitk.Image) -> sitk.Image:
   
   
    padded_image = sitk.Image(original_image.GetSize(), 
                             cropped_image.GetPixelID(),
                             cropped_image.GetNumberOfComponentsPerPixel())
    padded_image.SetOrigin(original_image.GetOrigin())
    padded_image.SetSpacing(original_image.GetSpacing())
    padded_image.SetDirection(original_image.GetDirection())
     
    padded_array = sitk.GetArrayFromImage(padded_image)
    padded_array.fill(0)
    padded_image = sitk.GetImageFromArray(padded_array)
    padded_image.CopyInformation(original_image)
    
    
    crop_origin = cropped_image.GetOrigin()
    start_index = original_image.TransformPhysicalPointToIndex(crop_origin)

    
    patch_size = cropped_image.GetSize()
    for z in range(patch_size[2]):
        for y in range(patch_size[1]):
            for x in range(patch_size[0]):
                
                orig_x = start_index[0] + x
                orig_y = start_index[1] + y
                orig_z = start_index[2] + z
                
                
                if (orig_x >= 0 and orig_x < original_image.GetSize()[0] and 
                    orig_y >= 0 and orig_y < original_image.GetSize()[1] and 
                    orig_z >= 0 and orig_z < original_image.GetSize()[2]):
                    padded_image[orig_x, orig_y, orig_z] = cropped_image[x, y, z]
    
    return padded_image

if __name__ == "__main__":
    ################################# 参数设定 ###################################
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Dataset_root = 'D:\LJX_Data\dose_calculation\\train-data\\full-test-data'
    Dataset_image_path = os.path.join(Dataset_root, 'ct')
    Dataset_map_path = os.path.join(Dataset_root, 'map')
    Dataset_label_path = os.path.join(Dataset_root, 'pos')
    torch.cuda.empty_cache()
    model_path = "./200.pth"
    
    in_channels = 3
    
    input_shape = [32,32,32]
    stride = sub_patch_size = [32,32,32]
    Muti_GPU = False
   
    ################################# 加载模型 ###################################
  
    model = myDoseNet.myDoseNet(spatial_dims=3, in_channels= in_channels, out_channels=1, features=(16, 32, 64, 128, 256, 32))
    
    if Muti_GPU:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    
    image_list = os.listdir(Dataset_image_path)
    for file_image in tqdm(image_list):
        with torch.no_grad():
            image_path = os.path.join(Dataset_image_path, file_image)
            map_path = os.path.join(Dataset_map_path, file_image[0:-6]+'map.nii')
            pos_path = os.path.join(Dataset_label_path, file_image[0:-6]+'pos.nii')
            origin_image_ = sitk.ReadImage(image_path)
            
            pos_ = sitk.ReadImage(pos_path)
            pos = get_max_intensity_coordinate(pos_)
            pos_ = crop_from_pos(pos, pos_)
            
            min = -1000
            max = 3000
            ww_filter = sitk.IntensityWindowingImageFilter()
            ww_filter.SetWindowMinimum(min)
            ww_filter.SetWindowMaximum(max)
            ww_filter.SetOutputMinimum(min)
            ww_filter.SetOutputMaximum(max)
            origin_image_ = ww_filter.Execute(origin_image_)    
            image_ = crop_from_pos(pos, origin_image_)  
           
            map_ = sitk.ReadImage(map_path)
            map_ = crop_from_pos(pos,map_)
           
            train_image = torch.FloatTensor(sitk.GetArrayFromImage(map_)).unsqueeze(0).to(device).unsqueeze(0)
            train_label = torch.FloatTensor(sitk.GetArrayFromImage(image_)).unsqueeze(0).to(device).unsqueeze(0)
            train_map = torch.FloatTensor(sitk.GetArrayFromImage(pos_)).unsqueeze(0).to(device).unsqueeze(0)
            train_input = torch.cat((train_image, train_label, train_map), dim=1)
            pred_labels = model(train_input)
                
            output = pred_labels.squeeze(0).squeeze(0)
            output = output.detach().cpu().numpy()         
 
            pred_label = np.array(output)
            print(pred_label.shape)
            
        pred_label_image = sitk.GetImageFromArray(pred_label)
        pred_label_image.SetOrigin(image_.GetOrigin())
        pred_label_image.SetSpacing(image_.GetSpacing())
        
        
        pred_label_image = pad_to_original_size(pred_label_image, origin_image_)
        
        # save_dir = os.path.join(Dataset_root, 'dose_crop')
        # os.makedirs(save_dir, exist_ok=True)
        # Dataset_pred_path = os.path.join(save_dir, str(file_image[:-6]) + 'dose_pred.nii')
        # print(Dataset_pred_path)
        # sitk.WriteImage(pred_label_image, Dataset_pred_path)

  


