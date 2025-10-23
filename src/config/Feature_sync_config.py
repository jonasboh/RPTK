import pandas as pd



def check_mirp_features_classes(features):
    """
    Count number of features containing mirp features
    """
    selected_features = []
    mirp_feature_classes=["img_data","diag","morph","loc","stat","ivh","ih","cm","rlm","szm","dzm","ngt","ngl"]
    mirp_feature_number=0
    
    for feature in features:
        for feature_class in mirp_feature_classes:
            if feature_class in feature:
                check, fitting_mirp = check_mirp_features(feature)
                if check:
                    if feature not in selected_features:
                        mirp_feature_number+=1
                        selected_features.append(fitting_mirp)

    return mirp_feature_number, selected_features

def check_mirp_features(feature):
    # IBSI/MIRP feature space
    # stat_mad_raw_intra_zscore

    img_data = ["settings_id", "modality", "config",  "noise_level", "noise_iter", "rotation_angle", "roi_randomise_iter", "roi_adapt_size", "translate_x", "translate_y", "translate_z","voxel_size", "roi"]
    
    diagnostics = ["img_dim_x_init_img", "img_dim_y_init_img", "img_dim_z_init_img", "vox_dim_x_init_img", "vox_dim_y_init_img", "vox_dim_z_init_img", "mean_int_init_img", "min_int_init_img",
                   "max_int_init_img", "int_map_dim_x_init_roi", "dint_map_dim_y_init_roi", "int_map_dim_z_init_roi", "int_bb_dim_x_init_roi", "int_bb_dim_y_init_roi", "int_bb_dim_z_init_roi",
                   "int_vox_dim_x_init_roi", "int_vox_dim_y_init_roi", "int_vox_dim_z_init_roi", "int_vox_count_init_roi", "int_mean_int_init_roi", "int_min_int_init_roi", "int_max_int_init_roi",
                   "mrp_map_dim_x_init_roi", "mrp_map_dim_y_init_roi", "mrp_map_dim_z_init_roi", "mrp_bb_dim_x_init_roi", "mrp_bb_dim_y_init_roi", "mrp_bb_dim_z_init_roi", "mrp_vox_dim_x_init_roi",
                   "mrp_vox_dim_y_init_roi", "mrp_vox_dim_z_init_roi", "mrp_vox_count_init_roi", "mrp_mean_int_init_roi", "mrp_min_int_init_roi", "mrp_max_int_init_roi", "img_dim_x_interp_img",
                   "img_dim_y_interp_img", "img_dim_z_interp_img", "vox_dim_x_interp_img", "vox_dim_y_interp_img", "vox_dim_z_interp_img", "mean_int_interp_img", "min_int_interp_img",
                   "max_int_interp_img", "int_map_dim_x_interp_roi", "int_map_dim_y_interp_roi", "int_map_dim_z_interp_roi", "int_bb_dim_x_interp_roi", "int_bb_dim_y_interp_roi",
                   "int_bb_dim_z_interp_roi", "int_vox_dim_x_interp_roi", "int_vox_dim_y_interp_roi", "int_vox_dim_z_interp_roi", "int_vox_count_interp_roi", "int_mean_int_interp_roi",
                   "int_min_int_interp_roi", "int_max_int_interp_roi", "mrp_map_dim_x_interp_roi", "mrp_map_dim_y_interp_roi", "mrp_map_dim_z_interp_roi", "mrp_bb_dim_x_interp_roi",
                   "mrp_bb_dim_y_interp_roi", "mrp_bb_dim_z_interp_roi", "mrp_vox_dim_x_interp_roi", "mrp_vox_dim_y_interp_roi", "mrp_vox_dim_z_interp_roi", "mrp_vox_count_interp_roi",
                   "mrp_mean_int_interp_roi", "mrp_min_int_interp_roi", "mrp_max_int_interp_roi", "img_dim_x_reseg_img", "img_dim_y_reseg_img", "img_dim_z_reseg_img", "vox_dim_x_reseg_img",
                   "vox_dim_y_reseg_img", "vox_dim_z_reseg_img", "mean_int_reseg_img", "min_int_reseg_img", "max_int_reseg_img", "int_map_dim_x_reseg_roi", "int_map_dim_y_reseg_roi",
                   "int_map_dim_z_reseg_roi", "int_bb_dim_x_reseg_roi", "int_bb_dim_y_reseg_roi", "int_bb_dim_z_reseg_roi", "int_vox_dim_x_reseg_roi", "int_vox_dim_y_reseg_roi",
                   "int_vox_dim_z_reseg_roi", "int_vox_count_reseg_roi", "int_mean_int_reseg_roi", "int_min_int_reseg_roi", "int_max_int_reseg_roi", "mrp_map_dim_x_reseg_roi",
                   "mrp_map_dim_y_reseg_roi", "mrp_map_dim_z_reseg_roi", "mrp_bb_dim_x_reseg_roi", "mrp_bb_dim_y_reseg_roi", "mrp_bb_dim_z_reseg_roi", "mrp_vox_dim_x_reseg_roi",
                   "mrp_vox_dim_y_reseg_roi", "mrp_vox_dim_z_reseg_roi", "mrp_vox_count_reseg_roi", "mrp_mean_int_reseg_roi", "mrp_min_int_reseg_roi", "mrp_max_int_reseg_roi"]
    
    morphology = ["volume", "vol_approx", "area_mesh", "morph_av", "comp_1", "comp_2", "sph_dispr", "sphericity", 
                  "asphericity", "com", "diam", "pca_maj_axis", "pca_min_axis", "pca_least_axis", "pca_elongation",
                 "pca_flatness", "vol_dens_aabb", "area_dens_aabb", "vol_dens_aee", "vol_dens_conv_hull", 
                  "area_dens_conv_hull", "integ_int", "moran_i", "geary_c"]

    local_intensity = ["peak_loc", "peak_glob"]

    intensity_based_statistics = ["mean", "var", "skew", "kurt", "median", "min", "p10", "p90", "max", "iqr", "range",
                                 "mad", "rmad", "medad", "cov", "qcod", "energy", "rms"]

    # dependent on the base_discretisation_bin_width in feature extraction --> postfix: fbs_w6.0
    intensity_histogram = ["mean", "var", "skew", "kurt", "median", "min", "p10", "p90",
                          "max", "mode", "iqr", "range", "mad", "rmad", "medad", "cov",
                          "qcod", "entropy", "uniformity", "max_grad", "max_grad_g", "min_grad", 
                           "min_grad_g"]

    # dependent on the ivh_discretisation_bin_width in feartre extraction
    intensity_volume_histogram = ["v10", "v90", "i10", "i90", "diff_v10_v90", "diff_i10_i90"]
    intensity_volume_histogram_mirp = ["v10", "v25", "v50", "v75", "v90", "i10", "i25" , "i50" , 
                                       "i75" , "i90" , "diff_v10_v90" , "diff_v25_v75" , 
                                       "diff_i10_i90" , "diff_i25_i75"]

    # dependent on the base_discretisation_bin_width and on glcm_spatial_method (dimensions to extract from) --> postfix d1_2d_avg_fbs_w6.0 d1_3d_avg_fbs_w6.0 d1_2d_s_mrg_fbs_w6.0 d1_3d_v_mrg_fbs_w6.0
    glcm = ["joint_max", "joint_avg", "joint_var", "joint_entr", "diff_avg", "diff_var", "diff_entr", 
            "sum_avg", "sum_var", "sum_entr", "energy", "contrast", "dissimilarity", "inv_diff", "inv_diff_norm", 
            "inv_diff_mom", "inv_diff_mom_norm", "inv_var", "cm_corr", "auto_corr", "clust_tend", "clust_shade", 
            "clust_prom", "info_corr1", "info_corr2"]

    # dependent on the base_discretisation_bin_width and on the glrlm_spatial_method
    glrlm = ["sre", "lre", "lgre", "hgre", "srlge", "srhge", "lrlge", "lrhge", "glnu", "glnu_norm", 
             "rlnu", "rlnu_norm", "r_perc", "gl_var", "rl_var", "rl_entr"]

    # dependent on the base_discretisation_bin_width and on the glszm_spatial_method
    glszm = ["sze", "lze", "lgze", "hgze", "szlge", "szhge", "lzlge", "lzhge", "glnu", "glnu_norm", "zsnu", "zsnu_norm", 
             "z_perc", "gl_var", "zs_var", "zs_entr"]

    # dependent on the base_discretisation_bin_width and on the gldzm_spatial_method
    gldzm = ["sde", "lde", "lgze", "hgze", "sdlge", "sdhge", "ldlge", "ldhge", "glnu", "glnu_norm", "zdnu", "zdnu_norm", 
             "z_perc", "gl_var", "zd_var", "zd_entr"]

    # dependent on the base_discretisation_bin_width and on the ngtdm_spatial_method
    ngtdm = ["coarseness", "contrast", "busyness", "complexity", "strength"]

    # dependent on the base_discretisation_bin_width and on the ngldm_spatial_method
    ngldm = ["lde", "hde", "lgce", "hgce", "ldlge", "ldhge", "hdlge", "hdhge", "glnu", "glnu_norm", "dcnu", "dcnu_norm", 
             "dc_perc", "gl_var", "dc_var", "dc_entr", "dc_energy"]
    
    mirp_features = img_data + diagnostics + morphology + local_intensity + intensity_based_statistics + intensity_histogram + intensity_volume_histogram + intensity_volume_histogram_mirp + glcm + glrlm + glszm + gldzm + ngtdm + ngldm
    
    for mirp in mirp_features:
        if mirp in feature:
            return True, feature
        
    return False, None

def check_pyradiomics_feature_class(features):
    """
    Count number of features containing pyradiomics features
    """
    
    selected_features = []
    pyradiomics_feature_classes=["diagnostics_Versions","diagnostics_Configuration","diagnostics_Image-original","diagnostics_Mask-original","diagnostics_Mask-resegmented","shape","firstorder","glcm","glrlm","glszm","gldm","ngtdm"]
    
    pyradiomics_feature_number=0
    
    for feature in features:
        for feature_class in pyradiomics_feature_classes:
            if feature_class in feature:
                check, fitting_py = check_pyradiomics_features(feature)
                if check:
                    if feature not in selected_features:
                        pyradiomics_feature_number+=1
                        selected_features.append(fitting_py)

    return pyradiomics_feature_number, selected_features

def check_pyradiomics_features(feature):
    # Pyradiomics Featurespace
    first_order = ["Energy", "TotalEnergy", "firstorder_Entropy", "Minimum", 
                            "10Percentile", "90Percentile", "Maximum", "Mean", 
                            "Median", "InterquartileRange", "Range", "MeanAbsoluteDeviation", 
                            "RobustMeanAbsoluteDeviation", "RootMeanSquared", "StandardDeviation", "Skewness",
                            "Kurtosis", "Variance", "Uniformity"]

    shape_3D = ["MeshVolume", "VoxelVolume", "SurfaceArea", "SurfaceVolumeRatio", "Sphericity",
                                  "Compactness1", "Compactness2", "SphericalDisproportion", "Maximum3Ddiameter",
                                 "Maximum2DDiameterSlice", "Maximum2DDiameterRow", "MajorAxisLength", " MinorAxisLength",
                                 "LeastAxisLength", "Elongation", "Flatness"]

    shape_2D = ["MeshSurface", "PixelSurface", "Perimeter", "PerimeterSurfaceratio", "Sphericity",
                                   "SphericalDisproportion", "Maximum2Ddiameter", "MajorAxisLength", "MinorAxisLength", 
                                    "Elongation"]

    glcm = ["Autocorrelation", "ClusterProminence", "ClusterShade", "ClusterTendency", "Contrast",
                                "Correlation", "DifferenceAverage", "DifferenceEntropy", "DifferenceVariance", "glcm_Id_", "Idm", 
                                "Idmn", "Idn", "Imc1", "Imc2", "InverseVariance", "JointAverage", "JointEnergy", 
                                 "JointEntropy", "MCC", "MaximumProbability", "SumAverage", "SumEntropy", "SumSquares"]

    glrlm = ["GrayLevelNonUniformity", "GrayLevelNonUniformityNormalized", "GrayLevelVariance",
                                 "HighGrayLevelRunEmphasis", "LongRunEmphasis", "LongRunHighGrayLevelEmphasis",
                                 "LongRunLowGrayLevelEmphasis", "LowGrayLevelRunEmphasis", "RunEntropy",
                                 "RunLengthNonUniformity", "RunLengthNonUniformityNormalized", "RunPercentage",
                                 "RunVariance", "ShortRunEmphasis", "ShortRunHighGrayLevelEmphasis", 
                                  "ShortRunLowGrayLevelEmphasis"]

    glszm = ["GrayLevelNonUniformity", "GrayLevelNonUniformityNormalized", "GrayLevelVariance",
                                 "HighGrayLevelZoneEmphasis", "LargeAreaEmphasis", "LargeAreaHighGrayLevelEmphasis",
                                 "LargeAreaLowGrayLevelEmphasis", "LowGrayLevelZoneEmphasis", "SizeZoneNonUniformity",
                                 "SizeZoneNonUniformityNormalized", "SmallAreaEmphasis", "SmallAreaHighGrayLevelEmphasis",
                                 "SmallAreaLowGrayLevelEmphasis", "ZoneEntropy", "ZonePercentage", "ZoneVariance"]

    ngtdm = ["Busyness", "Coarseness", "Complexity", "Contrast", "Strength"]

    gldm = ["DependenceEntropy", "DependenceNonUniformity", "DependenceNonUniformityNormalized",
                                "DependenceVariance", "GrayLevelNonUniformity", "GrayLevelVariance", "HighGrayLevelEmphasis",
                                "LargeDependenceEmphasis", "LargeDependenceHighGrayLevelEmphasis", "LargeDependenceLowGrayLevelEmphasis",
                                "LowGrayLevelEmphasis", "SmallDependenceEmphasis", "SmallDependenceHighGrayLevelEmphasis", 
                                "SmallDependenceLowGrayLevelEmphasis"]

    diagnostics = ["Image-original_Mean", "Image-original_Minimum", "Image-original_Maximum", "Image-original_Spacing",
                                        "Mask-original_BoundingBox", "Mask-original_VoxelNum", "Mask-original_VolumeNum", 
                                        "Mask-original_CenterOfMassIndex", "Mask-original_CenterOfMass", "Mask-original_Mean", 
                                        "Mask-original_Minimum", "Mask-original_Maximum","Mask-resegmented_BoundingBox", 
                                        "Mask-resegmented_VoxelNum", "Mask-resegmented_VolumeNum", "Mask-resegmented_CenterOfMassIndex", 
                                        "Mask-resegmented_CenterOfMass", "Mask-resegmented_Mean", "Mask-resegmented_Minimum", 
                                        "Mask-resegmented_Maximum"]
    
    pyradiomics_features = first_order + shape_3D + shape_2D + glcm + glrlm + glszm + ngtdm + gldm + diagnostics
    
    for py in pyradiomics_features:
        if py in feature:
            return True, feature
        
    return False, None

