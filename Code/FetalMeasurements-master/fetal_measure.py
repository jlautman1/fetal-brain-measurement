from os.path import join
import numpy as np
import nibabel as nib
import torch
import builtins
import sys
sys.path.append('/workspace/fetal-brain-measurement/Code/FetalMeasurements-master/SubSegmentation')
import gzip
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from lovasz import lovasz_softmax
from processing_utils import acc_no_bg

from fastai.basic_train import load_learner

#learn = load_learner(model_path, model_name)
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import os
import json
import textwrap
import fetal_seg
from SubSegmentation.lovasz import *
from SubSegmentation.processing_utils import *
from SubSegmentation.brain_segmentation_model import BrainSegmentationModel
import slice_select
from scipy import ndimage

import CBD_BBD
import msl

import re

# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):  # catches np.bool_, np.int_, etc.
            return obj.item()
        return super().default(obj)

#FD_RE = re.compile(
#            "Pat(?P<patient_id>[\d]+)_Se(?P<series>[\d]+)_Res(?P<res_x>[\d.]+)_(?P<res_y>[\d.]+)_Spac(?P<res_z>[\d.]+).nii")
FD_RE = re.compile(
    r"Pat(?P<patient_id>[\d]+)_Se(?P<series>[\d]+)_Res(?P<res_x>[\d.]+)_(?P<res_y>[\d.]+)_Spac(?P<res_z>[\d.]+)\.nii\.gz")


class FetalMeasure(object):
    def __init__(self, basedir="fetal-brain-measurement/Models",
                 braigseg_roi_model="22-ROI",
                 braigseg_seg_model="24-seg",
                 subseg_model='model-tfms.pkl',
                 sliceselect_bbd_model="23_model_bbd",
                 sliceselect_tcd_model="24_model_tcd"):
        self.fetal_seg = fetal_seg.FetalSegmentation(
            os.path.normpath(os.path.join(basedir, "Brainseg", "22-ROI")), None,
            os.path.normpath(os.path.join(basedir, "Brainseg", "24-seg")), None)
        print("model1 path: ", self.fetal_seg._model_path)
        print("model2 path: ", self.fetal_seg._model2_path)
        #self.subseg_learner = load_learner(os.path.join(basedir, "Subseg"), subseg_model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.subseg = BrainSegmentationModel(
        #    (160, 160),
        #    r"\\fmri-df4\projects\Mammilary_Bodies\BrainBiometry-IJCARS-Netanell\Models\Subseg",
        #    subseg_model,
        #    device=device
        #)
        model_dir = "/workspace/fetal-brain-measurement/Models/Subseg"
        print("subseg model before init: ", subseg_model)
        self.subseg = BrainSegmentationModel((160,160), model_dir, subseg_model, device=device)
        #this is the line that if documented it prodoces 6 output files
        
        
        #self.subseg = BrainSegmentationModel((160,160), os.path.join(basedir, "Subseg"), subseg_model, device=device)
        # self.sl_bbd = slice_select.SliceSelect(
        #     model_file="/home/netanell/work/FetalMeasurements/Models/Sliceselect/19_model_bbd",
        #     basemodel='ResNet50', cuda_id=0)
        # #model_file = "/home/netanell/work/research/slice_select/models/6/epoch0029_02-02_1249_choose_acc0.9678.statedict.pkl")
        # self.sl_tcd = slice_select.SliceSelect(
        #     model_file="/home/netanell/work/research/slice_select/models/7/epoch0029_02-02_1430_choose_acc0.9672.statedict.pkl", cuda_id=0)

        self.sl_bbd = slice_select.SliceSelect(
            model_file= os.path.normpath(os.path.join(basedir, "Sliceselect", "22_model_bbd")),
            basemodel='ResNet50', cuda_id=0)
        #model_file = "/home/netanell/work/research/slice_select/models/6/epoch0029_02-02_1249_choose_acc0.9678.statedict.pkl")
        self.sl_tcd = slice_select.SliceSelect(
            model_file= os.path.normpath(os.path.join(basedir, "Sliceselect", "25_model_tcd")),
            basemodel='ResNet50', cuda_id=0)
        print("Finished to load **************************")

    def _predict_nifti_subseg(self, img_data, filename):
        IMAGE_SIZE = (160, 160)
        images, min_ax, zeros, x_ax, y_ax = pre_processing(img_data, IMAGE_SIZE)
        segmentations_result = []
        zoomratio = min_ax/IMAGE_SIZE[0]
        for image in images:
            pred_class, pred_idx, outputs = self.subseg_learner.predict(image)
            segmentations_result.append(ndimage.zoom(pred_idx.data.squeeze(), zoomratio, order=0))



        seg_img = post_processing(segmentations_result, min_ax, zeros, x_ax, y_ax, filename)

    def execute(self, in_img_file, out_dir):
        ############################################################### model checking:##############################################################
        # print("checking the model: ")
        # p = "/workspace/fetal-brain-measurement/Models/Subseg/model-tfms.pkl"
        # print("Exists:",     os.path.exists(p))
        # print("File size:",  os.path.getsize(p), "bytes")

        # device = torch.device("cpu")
        # model = BrainSegmentationModel(
        # input_size=(160,160),
        # model_path="/workspace/fetal-brain-measurement/Models/Subseg",
        # model_name="model-tfms",
        # device=device
        # )
        # # If you get here without error, the .pkl at least parsed.
        # print(model.model)    # 
        # print("model printed, finished model checking, let's goooooQQQQQQQQQQQQQQQQQQQQQQQQQQ")
        # net = model.model.model  
        # dummy = torch.rand(1,3,160,160).to(device)
        # pred = net(dummy)
        # print("Output shape:", pred.shape)   #
        
        ############################################################## Not model checking any more########################################################################
        print("inside fetal measure execute printing input and out_dir: \n", in_img_file ,"\n",out_dir)
        # Full pipeline
        # in_img_file = "/media/df3-dafna/Netanell/DemoCode/Pat02_Se06_Res0.7422_0.7422_Spac5.nii"
        # out_dir = "/media/df3-dafna/Netanell/DemoCode/Outputs"

        elem_fname = os.path.basename(in_img_file) #takes the file name out of the path
        #pat_id, ser_num, res_x, res_y, res_z = [float(x) for x in FD_RE.findall(elem_fname)[0]]
        #match = FD_RE.search(os.path.basename(os.path.dirname(elem_fname)))
        #match = FD_RE.search(os.path.basename(os.path.dirname(in_img_file)))
        match = FD_RE.search(os.path.basename(in_img_file))

        if not match:
            raise ValueError(f"Filename format not recognized: {elem_fname}")
        pat_id, ser_num, res_x, res_y, res_z = [float(x) for x in match.groups()]

        metadata = {}
        metadata["InFile"] = in_img_file
        metadata["OutDir"] = out_dir
        metadata["Resolution"] = (res_x, res_y, res_z)
        metadata["SubjectID"] = pat_id
        metadata["Series"] = ser_num
		
        print("the metada is: ", metadata)
		
        # Prep image
        os.makedirs(out_dir, exist_ok=True)
        fn = os.path.basename(in_img_file) # the same as elem_fname
        elem_nii = nib.load(in_img_file) # creates a nifti file out of the file path
        # print("elem_nii is : ", elem_nii)
        elem_nii_arr = elem_nii.get_fdata() 
        # print("elem_nii_arr is : ", elem_nii_arr)
        # print("Volume stats:")
        # print("  shape:", elem_nii_arr.shape)
        # print("  min:", np.min(elem_nii_arr))
        # print("  max:", np.max(elem_nii_arr))
        # print("  unique:", np.unique(elem_nii_arr))
        input_axes = np.argsort(-np.array(elem_nii_arr.shape))
        #print("input axes: ", input_axes)
        nib_out = nib.Nifti1Image(elem_nii_arr.transpose(input_axes), affine=np.eye(4))
        #print("nib_out: ", nib_out)
        reorint_image_niifile =  os.path.normpath(os.path.join(out_dir, fn))
        #nib.save(nib_out, reorint_image_niifile)
        if reorint_image_niifile.endswith(".gz"):
            with gzip.open(reorint_image_niifile, 'wb') as f:
                f.write(nib_out.to_bytes())
        else:
            nib.save(nib_out, reorint_image_niifile)
        print("reorint_image_niifile is : ", reorint_image_niifile)
        #Debugging
        print("✅ Starting masking the brain...")
        self.fetal_seg.predict(reorint_image_niifile, out_dir)

        # ⚠️ Fix: Rename prediction_all.nii → prediction.nii.gz if needed
        pred_all = os.path.join(out_dir, "prediction_all.nii")
        pred_fix = os.path.join(out_dir, "prediction.nii.gz")
        if os.path.exists(pred_all) and not os.path.exists(pred_fix):
            os.rename(pred_all, pred_fix)
            print("🔁 Renamed prediction_all.nii → prediction.nii.gz")

        print("✅ Finished masking the brain")
        #return #jjjjjjjjjjjjjjj

        # First stage - Segmentation
        #self.fetal_seg.predict(reorint_image_niifile, out_dir)
        seg_file =  os.path.normpath(os.path.join(out_dir, "prediction.nii.gz"))
        # roi = nib.load(seg_file).get_fdata()
        # assert roi.sum() > 0, "ERROR: stage1 ROI mask is empty!"
        # print("🧪 Checking prediction.nii.gz contents...")
        # print("   ➤ prediction shape:", roi.shape)
        # print("   ➤ prediction unique values:", np.unique(roi))
        # print("   ➤ prediction max value:", np.max(roi))
        # print("   ➤ prediction nonzero voxel count:", np.count_nonzero(roi))
        #if out_dir.endswith(".nii.gz"):
        #    correct_out_dir = os.path.dirname(out_dir)
        #else:
        #    correct_out_dir = out_dir
        #subseg_file = os.path.join(correct_out_dir, "subseg.nii.gz")

        
        #subseg_file = os.path.normpath(out_dir)
        subseg_file =  os.path.normpath(os.path.join(out_dir, "subseg.nii.gz"))
        print("subseg_file: ", subseg_file)

        # Second stage - Slice select
        print("✅ Starting SL_TCD slice selection...")

        sl_tcd_result = self.sl_tcd.execute(img_file=reorint_image_niifile,
                                            seg_file=seg_file,visualize= True)
        print("sl_tcd_result: ", sl_tcd_result)
        print("✅ Finished SL_TCD slice selection")
        sl_tcd_slice = int(sl_tcd_result["prediction"].values[0])
        print("sl_tcd_result[prediction].values[0]", sl_tcd_result["prediction"].values[0])
        metadata["TCD_selection"] = sl_tcd_slice
        metadata["TCD_selectionValid"] = sl_tcd_result["isValid"].values[0]
        metadata["TCD_result"] = sl_tcd_result.prob_vec.tolist()
        print("slice_select_tcd: ", sl_tcd_slice)
        sl_bbd_result = self.sl_bbd.execute(img_file=reorint_image_niifile,
                                            seg_file=seg_file, )
        sl_bbd_slice = int(sl_bbd_result["prediction"].values[0])
        metadata["BBD_selection"] = sl_bbd_slice
        metadata["BBD_result"] = sl_bbd_result.prob_vec.tolist()
        metadata["BBD_selectionValid"] = sl_bbd_result["isValid"].values[0]

        data_cropped, fullseg = self.sl_tcd.get_cropped_elem(img_file=reorint_image_niifile,
                                                             seg_file=seg_file, )
        print("✅ fullseg unique values:", np.unique(fullseg))
        print("→ cropped data shape:",     data_cropped.shape)
        print("→ cropped data min/max:",   data_cropped.min(), data_cropped.max())
        print("→ cropped data nonzero voxels:", np.count_nonzero(data_cropped))
        print("→ fullseg (ROI) sum:",      fullseg.sum(), "out of", fullseg.size)
        #print("LOOOOOOOOOOOOOOOOOK3 data_cropped seems good? ", data_cropped )
        #print("LOOOOOOOOOOOOOOOOOK3 what about fulseg??  ", fullseg )
        
        data_cropped = data_cropped.transpose([1, 2, 0])
        fullseg = fullseg.transpose([1, 2, 0])
        #print("🧪 DEBUG: real subseg input stats")
        print("↳ shape:", data_cropped.shape)
        print("↳ dtype:", data_cropped.dtype)
        print("↳ min/max:", data_cropped.min(), data_cropped.max())
        print("↳ mean/std:", np.mean(data_cropped), np.std(data_cropped))
        print("↳ unique values (if small set):", np.unique(data_cropped)[:10])
        print("✅ post-transpose fullseg unique values:", np.unique(fullseg))
        print("✅ post-transpose fullseg dtype:", fullseg.dtype)
        #print("LOOOOOOOOOOOOOOOOOK4!! data_cropped seems good after? ", data_cropped )
        #print("LOOOOOOOOOOOOOOOOOK4!! what about fulseg after??  ", fullseg )
        nii_data_cropped = nib.Nifti1Image(data_cropped, affine=np.eye(4))
        nib.save(nii_data_cropped,  os.path.normpath(os.path.join(out_dir, "cropped.nii.gz")))

        nii_seg_cropped = nib.Nifti1Image(fullseg.astype(float), affine=np.eye(4))
        print("✅ seg.cropped.nii.gz voxel sum (after cast to float):", fullseg.astype(float).sum())
        nib.save(nii_seg_cropped,  os.path.normpath(os.path.join(out_dir, "seg.cropped.nii.gz")))
        
        # Third stage - Sub segmentaion

        #self._predict_nifti_subseg(data_cropped.copy(), subseg_file)
        print("📊 Cropped input range BEFORE subseg:", data_cropped.min(), data_cropped.max())
        self.subseg.predict_nifti(data_cropped.copy().astype(np.float32), subseg_file, tta=True)
        print("seeing the subseg_file: ", subseg_file)
        subseg = nib.load(subseg_file).get_fdata()
        print("✅ subseg shape:", subseg.shape)
        print("✅ subseg unique values:", np.unique(subseg))
        print("✅ subseg max value:", np.max(subseg))
        print("✅ subseg voxel sum:", np.sum(subseg))
        #print("subseg: ", subseg)
        if np.max(subseg) == 0:
            print(f"[ERROR] Subsegmentation failed for {in_img_file}. Skipping.")
            #return metadata
        #print("LOOOOOOOOOOOOOOOOOK4 seems good subseg") #REMOVE LATER 
        # Fourth stage - MSL
        print("before MSL")
        msl_p_planes = msl.find_planes(data_cropped, subseg)
        print("middle of MSL")
        msl_p_points = msl.findMSLforAllPlanes(data_cropped, subseg, msl_p_planes)
        #print("LOOOOOOOOOOOOOOOOOK ",msl_p_planes) #REMOVE LATER 
        #print("LOOOOOOOOOOOOOOOOOK2 ",msl_p_points) #REMOVE LATER 
        #print(msl.__file__) I checked it and it is imported properly
        print("after MSL")
        metadata["msl_planes"] = msl_p_planes
        metadata["msl_points"] = msl_p_points

        # Fifth - Measuring
        print("\n===== DEBUG: About to look up BBD slice in MSL results =====")
        print("    sl_tcd_slice =", sl_tcd_slice)
        print("    sl_bbd_slice =", sl_bbd_slice)
        print("    len(msl_p_points)  =", len(msl_p_points))
        print("    valid indices [0 ..", len(msl_p_points)-1, "]")
        # BBD + CBD
        CBD_min_th = (subseg.shape[0] / 4)
        p_u, p_d, _ = msl_p_points[sl_bbd_slice]
        CBD_measure, CBD_left, CBD_right = CBD_BBD.CBD_points(subseg[:, :, sl_bbd_slice] > 0,
                                                                   p_u.astype(int),
                                                                   p_d.astype(int), res_x, res_y,
                                                                  CBD_min_th)
        metadata["cbd_points"] = (CBD_left, CBD_right)
        metadata["cbd_measure_mm"] = CBD_measure
        BBD_measure, BBD_left, BBD_right, BBD_valid = CBD_BBD.BBD_points(data_cropped[:, :, sl_bbd_slice], CBD_left, CBD_right,
                                                             p_u.astype(int), p_d.astype(int),
                                                  res_x, res_y)
        metadata["bbd_points"] = (BBD_left, BBD_right)
        metadata["bbd_measure_mm"] = BBD_measure
        metadata["bbd_valid"] = BBD_valid

        #print(np.linalg.norm(CBD_left - CBD_right) * res_x, np.linalg.norm(BBD_left - BBD_right) * res_x)
        plt.figure()
        plt.imshow(data_cropped[:, :, sl_bbd_slice])
        cbd = np.stack([CBD_left, CBD_right]).T
        plt.plot(cbd[1, :], cbd[0, :], 'r')
        plt.savefig( os.path.normpath(os.path.join(out_dir, "cbd.png")))
        plt.figure()
        plt.imshow(data_cropped[:, :, sl_bbd_slice])
        bbd = np.stack([BBD_left, BBD_right]).T
        plt.plot(bbd[1, :], bbd[0, :], 'b-')
        plt.savefig( os.path.normpath(os.path.join(out_dir, "bbd.png")))

        # TCD
        plt.figure()
        p_u, p_d, _ = msl_p_points[sl_tcd_slice]
        TCD_measure, TCD_left, TCD_right, TCD_valid = CBD_BBD.TCD_points(subseg[:, :, sl_tcd_slice] == 2., p_u.astype(int), p_d.astype(int),
                                                 res_x, res_y)
        metadata["tcd_valid"] = TCD_valid
        metadata["tcd_points"] = (TCD_left, TCD_right)
        metadata["tcd_measure_mm"] = TCD_measure
        plt.imshow(data_cropped[:, :, sl_tcd_slice])
        if TCD_measure is not None:
            tcd = np.stack([TCD_left, TCD_right]).T
            plt.plot(tcd[1, :], tcd[0, :], 'k-')
            plt.savefig( os.path.normpath(os.path.join(out_dir, "tcd.png")))

        # Brain Volume Calc

        metadata["brain_vol_voxels"] = float(np.sum(fullseg > .5))
        metadata["brain_vol_mm3"] = float(np.sum(fullseg > .5) * res_x * res_y * res_z)
        # Build a short summary based on metadata
        summary = {
            "brain_volume_mm3": builtins.round(float(metadata["brain_vol_mm3"])),
            "CBD (mm)": builtins.round(float(metadata.get("cbd_measure_mm", 0))),
            "BBD (mm)": builtins.round(float(metadata.get("bbd_measure_mm", 0))),
            "TCD (mm)": builtins.round(float(metadata.get("tcd_measure_mm", 0))),
            "TCD valid": bool(metadata.get("tcd_valid", False)),
            "BBD valid": bool(metadata.get("bbd_valid", False)),
            "Volume path": metadata["InFile"]
        }

        # Append summary to metadata
        metadata["summary"] = summary
        # Dump metadata
        with open(os.path.normpath(os.path.join(out_dir, 'data.json')), 'w') as fp:
            json.dump(metadata, fp, cls=NumpyEncoder,indent=4)

        # ---- begin PDF report generation (with header + footer) ----
       

        report_path = os.path.join(out_dir, 'report.pdf')
        with PdfPages(report_path) as pdf:
            fig = plt.figure(figsize=(8.5, 11))
            # make header taller to give more space, bump hspace
            gs = gridspec.GridSpec(4, 2, figure=fig,
                                height_ratios=[1.7, 3, 3, 3],
                                hspace=1.0, wspace=0.3)

            # Header
            ax_hdr = fig.add_subplot(gs[0, :])
            ax_hdr.axis('off')
            pid = int(metadata['SubjectID'])
            ser = int(metadata['Series'])
            rx, ry, rz = metadata['Resolution']
            ax_hdr.text(0.5, 0.80,
                        f"Fetal brain measurements analysis for patient #{pid}",
                        ha='center', va='center',
                        fontsize=18, fontweight='bold')
            ax_hdr.text(0.5, 0.50,
                        f"Series: {ser}    Voxel size: {rx:.4g}×{ry:.4g}×{rz:.4g} mm",
                        ha='center', va='center',
                        fontsize=12)

            # Row‐drawing helper
            def draw_row(r, imgfile, label):
                im_ax = fig.add_subplot(gs[r, 0])
                im_ax.imshow(plt.imread(os.path.join(out_dir, imgfile)))
                im_ax.axis('off')

                tx_ax = fig.add_subplot(gs[r, 1])
                tx_ax.axis('off')
                tx_ax.text(0, 0.5, label,
                        ha='left', va='center',
                        fontsize=14, family='monospace')

            # CBD (measured on the BBD slice)
            draw_row(1, 'cbd.png',
                    f"CBD (mm): {metadata['cbd_measure_mm']:.2f}\n"
                    f"(slice #{metadata['BBD_selection']})")

            # BBD
            draw_row(2, 'bbd.png',
                    f"BBD (mm): {metadata['bbd_measure_mm']:.2f}\n"
                    f"(slice #{metadata['BBD_selection']})")

            # TCD
            draw_row(3, 'tcd.png',
                    f"TCD (mm): {metadata['tcd_measure_mm']:.2f}\n"
                    f"(slice #{metadata['TCD_selection']})")

            # Footer: only brain volume
            fig.text(0.05, 0.02,
                    f"Brain volume (mm³): {metadata['brain_vol_mm3']:.0f}",
                    fontsize=12)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        # ---- end PDF report generation ----


        return metadata
