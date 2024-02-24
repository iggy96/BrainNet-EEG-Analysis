from ds_helper import*
import sys
sys.path.insert(1, '/Users/joshuaighalo/Documents/GitHub/eegDementia/groupingScans')
import seperator_11 as sep


missen_scans = [152,280,409]
clinic_data = 'lp_clinical.xlsx'
scan_info = 'Scan_details.xlsx'
dataPath = '/Users/joshuaighalo/Downloads/brainNet_datasets/laurel_place/cleaned_dataset/'
numpy_dir = '/Users/joshuaighalo/Documents/GitHub/eegDementia/MLOps/dementiaStages/classified_scans/'


noB_1,no6_1,no12_1,noB_2,no6_2,no12_2 = sep.params(mmse_scale=[25,30],null_scans=missen_scans,clinical_filename=clinic_data,scaninfo_filename=scan_info,search_path=dataPath)
mildB_1,mild6_1,mild12_1,mildB_2,mild6_2,mild12_2 = sep.params(mmse_scale=[20,24],null_scans=missen_scans,clinical_filename=clinic_data,scaninfo_filename=scan_info,search_path=dataPath)
modB_1,mod6_1,mod12_1,modB_2,mod6_2,mod12_2 = sep.params(mmse_scale=[13,19],null_scans=missen_scans,clinical_filename=clinic_data,scaninfo_filename=scan_info,search_path=dataPath)
sevB_1,sev6_1,sev12_1,sevB_2,sev6_2,sev12_2 = sep.params(mmse_scale=[0,12],null_scans=missen_scans,clinical_filename=clinic_data,scaninfo_filename=scan_info,search_path=dataPath)
mildmodB_1 = mildB_1 + modB_1
mildmod6_1 = mild6_1 + mod6_1
mildmod12_1 = mild12_1 + mod12_1


qBND = removeBadRawEEGs(noB_1,1.0,'No Dementia | Baseline | Run 1: ',dataPath)
q6ND = removeBadRawEEGs(no6_1,1.0,'No Dementia | 6 Months | Run 1: ',dataPath)
q12ND = removeBadRawEEGs(no12_1,1.0,'No Dementia | 12 Months | Run 1: ',dataPath)
qBMIDMOD = removeBadRawEEGs(mildmodB_1,1.0,'Mild-Moderate Dementia | Baseline | Run 1: ',dataPath)
q6MIDMOD = removeBadRawEEGs(mildmod6_1,1.0,'Mild-Moderate Dementia | 6 Months | Run 1: ',dataPath)
q12MIDMOD = removeBadRawEEGs(mildmod12_1,1.0,'Mild-Moderate Dementia | 12 Months | Run 1: ',dataPath)
qBSD = removeBadRawEEGs(sevB_1,1.0,'Severe Dementia | Baseline | Run 1: ',dataPath)
q6SD = removeBadRawEEGs(sev6_1,1.0,'Severe Dementia | 6 Months | Run 1: ',dataPath)
q12SD = removeBadRawEEGs(sev12_1,1.0,'Severe Dementia | 12 Months | Run 1: ',dataPath)

# export data as numpy binary files
np.save(numpy_dir+'baselineNoDementia.npy',qBND),np.save(numpy_dir+'6NoDementia.npy',q6ND),np.save(numpy_dir+'12NoDementia.npy',q12ND)
np.save(numpy_dir+'baselineMildModDementia.npy',qBMIDMOD),np.save(numpy_dir+'6MildModDementia.npy',q6MIDMOD),np.save(numpy_dir+'12MildModDementia.npy',q12MIDMOD)
np.save(numpy_dir+'baselineSevereDementia.npy',qBSD),np.save(numpy_dir+'6SevereDementia.npy',q6SD),np.save(numpy_dir+'12SevereDementia.npy',q12SD)

