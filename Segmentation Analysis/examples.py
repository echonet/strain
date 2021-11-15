"""
Old code for variance plot
"""




df = pd.DataFrame({"Filenames":files,"Strain":strains})
#df.to_csv(os.path.join(basic_dir,'Variance Plot.csv'),index=False)

df = pd.DataFrame({"Filenames":files,"Strain":strains})
df = df[~(df.Strain=='[]')].reset_index()

def get_mean(string):
    x = string#get_arr(string)
    
    return np.mean(x[1:-1])
def get_arr(string):
    #x = string.split('[')[1].split(']')[0].split(',')
    total = []
    for i in string:
        total.append(float(i))
    total.sort()
    return total
def outliers_and_insiders(x):
    #if len(x)<3:
    #    return [x[0],x[-1]], []
    regular = [x[1],x[-2]]
    outliers = [x[0],x[-1]]
    #for i in x:
    #    if i<regular[0] or i>regular[-1]:
    #        outliers.append(i)
    return regular,outliers

df['mean_strain'] = df.Strain.apply(get_mean)

sorted_df = df[['Filenames','Strain','mean_strain']].sort_values('mean_strain').dropna().reset_index(drop=True)

plt.figure(figsize=(15,10))
plt.plot(sorted_df.index[10:],sorted_df.mean_strain[10:],color = 'blue')
plt.title("Strain Distribution plot")

# Plot Lines
points = []
missing = 0
rangey = []
for i in range(10,len(sorted_df.index)):
    x = get_arr(sorted_df.Strain[i])
    missing = 0
    if len(x)>=2:
        regular, outliers = outliers_and_insiders(x)
        # print(x,regular,outliers)
        #print(len(x))
        
        plt.plot([i-missing,i-missing],[regular[0],regular[-1]],linewidth=2,alpha=0.1,color='blue')
        if len(x)>3:
            rangey.append(regular[-1]-regular[0])
        for k in x:
            #if k > max([min(x),regular[0]]) and k < min([max(x),regular[1]]):
            plt.scatter(i,k,color='blue',s=0.5)
    else:
        missing+=1
    points.append(len(x))
plt.savefig(os.path.join(basic_dir,'Variance Plot.png'))




## To produce band altman limits, plot the 95% confidence interval of the mean difference