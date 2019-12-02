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
