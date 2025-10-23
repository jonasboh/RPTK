from multiprocessing import Pool
import numpy as np
import os
import SimpleITK as sitk
import argparse
import tqdm


class Resampler:
    """
    Resample nifti Images or Segmentations to 1mm x 1mm x 1mm.
    """
    def __init__(self,
                 input_folder: str = "",  # folder with nifties
                 output_folder: str = "",  # folder for output (resamples files)
                 input_file: str = "",  # single file to resample
                 input_files=None,
                 segmentation: bool = True,  # if True, use nearest neighbor interpolation (for segmentations)
                 # if False, use linear interpolation (for scans)
                 desired_sampling=None,  # isoptropic sampling (default 1x1x1)
                 n_cpus: int = 1,  # number of cores to use
                 logger=None,
                 self_optimize: bool = False,
                 target:str = "Files"):

        if desired_sampling is None:
            desired_sampling = [1.0]
        if input_files is None:
            input_files = []

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.input_file = input_file
        self.segmentation = segmentation
        self.desired_sampling = desired_sampling
        self.n_cpus = n_cpus
        self.resampled_files = [] # List for resampled files
        self.logger = logger
        self.self_optimize = self_optimize
        self.target = target

        if self.input_folder != "":
            # get files fro input folder
            self.nifties = [os.path.join(path, name) for path, subdirs, files in os.walk(self.input_folder) for name in files if
                       name.endswith(".nii.gz")]
        elif len(input_files) > 0:
            self.nifties = input_files
        else:
            self.nifties = [self.input_file]

    def resample_image_with_spacing(self, image, interpolation_type, desired_sampling=None):
        """
        Read image and create a downsampled version according to DownsamplingFactor and interpolation method
        interpolation_type= True for Segmentation and interpolation_type=False for scans
        param image: image to resample
        param interpolation_type: True for Segmentation and interpolation_type=False for scans
        param desired_sampling: isotropic desired sampling (default: 1mm)
        out: resampled image
        """

        # get image properties and set new size and (isotropic) spacing
        if desired_sampling is None:
            desired_sampling = [1.0]

        input_spacing = np.array(image.GetSpacing())
        image_size = np.array(image.GetSize())
        
        # isotropic resampling
        if len(desired_sampling) == 1:
            desired_sampling = tuple(np.array(3 * [desired_sampling[0]]))
        # only optimized for slice thickness estimation on the z axis
        elif(self.self_optimize):
            desired_sampling = tuple(input_spacing[0], input_spacing[1], desired_sampling[0])
        # other resampling
        else:
            desired_sampling = tuple(np.array(desired_sampling))
            
        output_image_size = np.floor((input_spacing / desired_sampling) * image_size)
        output_image_size = np.array(np.round(output_image_size).astype(np.uint32)).tolist()

        # Resample filter to isotropic scaling
        itk_resampler = sitk.ResampleImageFilter()
        if interpolation_type:
            itk_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            itk_resampler.SetInterpolator(sitk.sitkBSpline)
            

        itk_resampler.SetDefaultPixelValue(0)
        itk_resampler.SetOutputOrigin(image.GetOrigin())
        itk_resampler.SetOutputDirection(image.GetDirection())
        itk_resampler.SetOutputSpacing(desired_sampling)
        itk_resampler.SetSize(output_image_size)

        image_new = itk_resampler.Execute(image)
        
        return image_new

    def call_resample_image(self, in_file):
        """
        Resample a single file.
        param in_file: file to resample
        out: resampled file
        """

        if not os.path.exists(os.path.join(self.output_folder, 
                                           os.path.basename(in_file)[:-len(".nii.gz")] + "_resampled.nii.gz")):
            
            # if image is not resampled already
            if "_resampled" not in str(os.path.basename(in_file)):

                # Read
                img = sitk.ReadImage(in_file)
                
                # Resample
                resampled = self.resample_image_with_spacing(image=img, interpolation_type=self.segmentation, desired_sampling=self.desired_sampling)
                
                # Write
                sitk.WriteImage(resampled, os.path.join(self.output_folder, os.path.basename(in_file)[:-len(".nii.gz")] + "_resampled.nii.gz"), useCompression=True)
        
        #self.resampled.append(os.path.join(self.output_folder, os.path.basename(in_file)[:-len(".nii.gz")] + "_resampled.nii.gz"))

    def exe(self):
        """
        Resample all files in input folder to 1mm x 1mm x 1mm.
        Main function for resampling.
        """
        self.logger.info("Resampling files to " + str(self.desired_sampling) + " mm")

        self.logger.info("Start Resampling " + str(len(self.nifties)) + " files")

        # create output folder if not exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        if len(self.nifties) > 1:
            
            with Pool(self.n_cpus) as pool:
                for _ in tqdm.tqdm(pool.imap_unordered(self.call_resample_image, self.nifties), total=len(self.nifties), desc="Resample " + self.target):
                    pass
                
            #p = Pool(self.n_cpus)
            #p.map(self.call_resample_image, self.nifties)
            #p.close()
            #p.join()

        else:
            self.call_resample_image(self.nifties[0])

        # check for not resampled files and resample them
        for nifti in tqdm.tqdm(self.nifties, desc="Check Completeness of Resampled " + self.target):
            if "_resampled" not in str(os.path.basename(nifti)):
                if not os.path.exists(os.path.join(self.output_folder, os.path.basename(nifti)[:-len(".nii.gz")] + "_resampled.nii.gz")):
                    self.call_resample_image(nifti)
        
        #return self.resampled_files
