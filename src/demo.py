import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from task import METAVideos as METAVideos
from scipy.cluster.hierarchy import dendrogram, linkage

'''how to use'''
np.random.seed(0)
min_len = 30
holdout_llcid = 0
exlude_multi_chapter = 1
verbose = 0
meta = METAVideos(holdout_llcid=holdout_llcid, min_len=min_len, exlude_multi_chapter=exlude_multi_chapter, verbose=verbose)

video_id = '4.4.4'
print(type(meta.data))
print(meta.data.keys())
print(np.shape(meta.data[video_id]))
print(meta.data[video_id])

print(type(meta.data_llc))
print(meta.data_llc.keys())
print(np.shape(meta.data_llc[video_id]))
print(meta.data_llc[video_id])
meta.get_video(video_id)

'''how to use sample_event_info and get_data'''
X_list, llc_name_list = meta.get_data(to_torch=True)
print(len(X_list))
print(len(llc_name_list))


# visualize
for i in range(len(llc_name_list)):
    llc_name = llc_name_list[i]
    llc_id = meta.llc2llcid[llc_name]
    hlc2hlcid = meta.llc2hlcid[llc_name]
    print(f'%3d %3d\t{llc_name}'%(hlc2hlcid, llc_id))

'''low level event class frequency'''
meta.show_llc_frequency()

'''plot hier'''
# remove multi-chapter action (mca)
mask_rm_mca = np.array([v for v in meta.llc2hlcid.values()]) !=0
mean_pattern = meta.llc_mean_pattern[mask_rm_mca]
yticklabels = np.array(meta.all_llc)[mask_rm_mca]
metric = 'correlation'
meta.make_sns_clustermap(mean_pattern, yticklabels, metric)

linked = linkage(mean_pattern, method='average', metric=metric)
f, ax  = meta.make_dendrogram(linked, yticklabels)

'''show RDM '''
f, ax = plt.subplots(1,1, figsize=(5,4))
sns.heatmap(np.corrcoef(meta.llc_mean_pattern), square=True, ax=ax)
ax.set_xlabel('low level event type')
ax.set_ylabel('low level event type')

'''Transition structure '''
cpal = sns.color_palette('colorblind', n_colors=5)
cpal = np.roll(cpal, shift=1, axis=0)

llc2ordllcid = {}
for i in np.arange(0, 5, 1):
    mask = i == np.array(list(meta.llc2hlcid.values()))
    llc_hlc_i = np.array(meta.all_llc)[mask]
    for k in range(len(llc_hlc_i)):
        llc2ordllcid[llc_hlc_i[k]] = len(llc2ordllcid)

transition_matrix = np.zeros((len(llc2ordllcid), len(llc2ordllcid)))
for k in meta.data_llc.keys():
    for t in range(len(meta.data_llc[k])-1):
        this_event = meta.data_llc[k][t]
        next_event = meta.data_llc[k][t+1]
        this_event_id = llc2ordllcid[this_event]
        next_event_id = llc2ordllcid[next_event]
        transition_matrix[this_event_id, next_event_id]+=1

f, ax = plt.subplots(1,1, figsize=(20, 20))
sns.heatmap(transition_matrix,ax=ax, square=True, cmap='bone')
ax.set_title('Low level event transition counts')
ax.set_yticks(np.arange(len(llc2ordllcid))+.5)
ax.set_yticklabels(list(llc2ordllcid.keys()), rotation=0)
ax.set_xticklabels([])
for ytick in ax.get_yticklabels():
    ytick.set_color(cpal[meta.llc2hlcid[ytick.get_text()]])
