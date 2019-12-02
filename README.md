# basic
basic
boolean = any(num_clm_policy_up[grp_col].duplicated())
def cat_col_top(df,col):
    if len(df[[col]].groupby([col]).size())>11:
        df1 = df[[col]].groupby([col]).size().nlargest(10).reset_index(name='top10')
        lst=df1[col]
        df.loc[~df[col].isin(lst),col]='Others'
    return df
    
    for i in range(len(detl_cat_col)):
    #print(detl_cat_col[i])
    detail_policy=cat_col_top(detail_policy,detl_cat_col[i])
    #print(detail_policy.shape)
    
    
    # This function aggregates the vector 
def agg_all_col(detail_df,tmp_var,base_df,grp_col):
    
    terr=detail_df[tmp_var].unique()# Get unique values of the category across the table
    terr=np.asarray(terr)# Convert teh list to array

    # This section will takes values of the column and convert to array
    token=detail_df[tmp_var]
    token=np.asarray(token)
    
    # This section will takes values of the column and convert to unique columns, 
    # one column to multiple column based on unique values
    sentence_vectors = []

    for i in range(len(token)):
        sent_vec = []
        for j in range(len(terr)):
            if token[i] ==terr[j]:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors.append(sent_vec)

    #Convert the array to pandas dataframe
    tmp_df=pd.DataFrame(sentence_vectors,columns=list(terr))
    tmp_df.columns = [tmp_var+str(col)  for col in tmp_df.columns]

    tmp_df=tmp_df.join(detail_df[grp_col])# get the group columns

    # Remove the group columns from the list of columns we want to aggreage
    attribs_mod = list(tmp_df.columns)
    for i in range(len(grp_col)):
        if grp_col[i] in attribs_mod: 
            #some_list.remove(thing)
            attribs_mod.remove(grp_col[i])   

    # This is the key section, this will aggregate the column and join to the base policy column
    for i in range(len(attribs_mod)):
        #print(attribs_mod[i])
        tmp_df_new=tmp_df.groupby(grp_col,as_index=False)[attribs_mod[i]].aggregate('mean')
        base_df= base_df.merge(tmp_df_new, how='left',left_on = grp_col,right_on = grp_col)
    return base_df
    
    for i in range(len(detl_cat_col)):
    print(detl_cat_col[i])
    num_clm_policy_up=agg_all_col(detail_policy,detl_cat_col[i],num_clm_policy_up,grp_col)
    print(num_clm_policy_up.shape)
    
    
    #model_master_data_pd = create_dummies(model_master_data_pd)
import pandas as pd
import numpy as np

def create_dummies(master_data_pd,ignr_var,scoring=0):
        
    dat_cols = master_data_pd.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    
    num_attribs = list(master_data_pd.select_dtypes(include=np.number))
    cat_attribs = list(master_data_pd.select_dtypes(exclude=np.number))
    cat_attribs_mod = cat_attribs.copy()
    for i in range(len(ignr_var)):
        if ignr_var[i] in cat_attribs_mod: 
        #some_list.remove(thing)
            cat_attribs_mod.remove(ignr_var[i])   
        #print(ignr_var[i])
        #cat_attribs = [x for x in cat_attribs if x not in dat_cols]
    
    cat_attribs = [x for x in cat_attribs if x not in dat_cols]
    
    if scoring==1:
        master_data_pd_mod = pd.concat([master_data_pd.drop(cat_attribs + dat_cols, axis=1),
                                    pd.get_dummies(master_data_pd[cat_attribs_mod], drop_first=True)]
                                   , axis=1
                                  )
    else:
        master_data_pd_mod = pd.concat([master_data_pd.drop(cat_attribs_mod + dat_cols, axis=1),
                                    pd.get_dummies(master_data_pd[cat_attribs_mod], drop_first=True)]
                                   , axis=1
                                  )
    return master_data_pd_mod
    
    
    from matplotlib import pyplot as plt
import numpy as np

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def dual_axis_v2(df,feature):
   
    #feature, onlevel, y variable, ID
    
    tmp_df=df[[feature,'On-Level','Loss_incurred_ALAE_lessCat','ID']]
    tmp_df[feature].fillna("ZZ_Not Available", inplace = True) 
    tmp_df=(pd.DataFrame(tmp_df.groupby(feature, as_index=False).
                         agg({'On-Level': "sum",
                              "Loss_incurred_ALAE_lessCat": "sum",
                              "ID":"count"}
                            ))
           )
    tmp_df['Loss_Ratio']=tmp_df['Loss_incurred_ALAE_lessCat']/tmp_df['On-Level']
    tmp_df['Distribution']=tmp_df['ID']/len(df)

    sns.set(style="white", rc={"lines.linewidth": 3})

    fig, ax1 = plt.subplots(figsize=(15,8))
    ax2 = ax1.twinx()
    
    sns.barplot(x=tmp_df[feature],
            y=tmp_df['Loss_Ratio'], 
            color='#004488',
            ax=ax1)
    
    sns.lineplot(x=tmp_df[feature], 
             y=tmp_df['Distribution'],
             color='r',
            marker="o",
             ax=ax2)
    
    show_values_on_bars(ax2)
    show_values_on_bars(ax1)
    
#plt.legend(title='Loss Ratio and Frequency Distribution', loc='upper right', labels=['Loss Ratio', 'Distribution'])
    ax1.set_title('Adjusted Loss Ratio (Incurred Loss ALAE (less cat)/On-Level) and #of Records Distribution')

    plt.show()
    sns.set()
    return None


def dual_axis_cat(df,feature):    
    
    tmp_df=df[[feature,'On-Level','Loss_incurred_ALAE_lessCat','ID']]
    #feature=feature+str('bin')
    #grp=tmp_df[feature].max()/10
    tmp_df[feature+str('_bin')]=pd.cut(df[feature], bins=10,precision=0)
    tmp_df[feature+str('_bin')]= tmp_df[feature+str('_bin')].astype(str) 
    tmp_df[feature+str('_bin')].fillna("ZZ_Not Available", inplace = True) 
    tmp_df=pd.DataFrame(tmp_df.groupby(feature+str('_bin'), as_index=False).agg({'On-Level': "sum",'Loss_incurred_ALAE_lessCat': "sum","ID":"count"}))
    tmp_df['Loss_Ratio']=tmp_df['Loss_incurred_ALAE_lessCat']/tmp_df['On-Level']
    tmp_df['Distribution']=tmp_df['ID']/len(df)

    sns.set(style="white", rc={"lines.linewidth": 3})
    fig, ax1 = plt.subplots(figsize=(20,8))
    ax2 = ax1.twinx()
    
    s1=sns.barplot(x=tmp_df[feature+str('_bin')],
            y=tmp_df['Loss_Ratio'], 
            color='#004488',
            ax=ax1)
    
    s2=sns.lineplot(x=tmp_df[feature+str('_bin')],
             y=tmp_df['Distribution'],
             color='r',
            marker="o",
             ax=ax2)
    
    s1.set_xticklabels(
    s1.get_xticklabels(), 
    rotation=90, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='large'
    )
    
    show_values_on_bars(ax2)
    show_values_on_bars(ax1)

#plt.legend(title='Loss Ratio and Frequency Distribution', loc='upper right', labels=['Loss Ratio', 'Distribution'])
    ax1.set_title('Adjusted Loss Ratio (Incurred Loss ALAE (less cat)/On-Level) and #of Records Distribution-'+feature+str('_bin'))
    
    plt.show()
    sns.set()
