import pandas as pd
import numpy as np

df = pd.read_excel('ThereseStats.xlsx')

def hms_to_hours(s):
    h,m,sec = map(float, s.split(':'))
    return h + m/60 + sec/3600

df['Hours'] = df['Hours needed to achieve 4500 rows of fine-tuning data (using one RTX 6000 Pro)'].apply(hms_to_hours)
df['Hours_inv'] = 1/df['Hours']

metrics = ['Success % (code blocks that compiled and ran)',
           'Space freed of total space',
           'Average space taken over per iteration',
           'Hours_inv']

stds = {m: df[m].std() for m in metrics}

out2 = []
for i in range(1, len(df)):
    row = {'Model': df.loc[i, 'Model Name']}
    zsum = 0
    w = 1/5
    for m in metrics:
        zd = (df.loc[i, m] - df.loc[i-1, m]) / stds[m]
        row[m.replace(' ', '_') + '_zΔ'] = zd
        zsum += zd * w
    row['ImprovementScore_unweighted'] = zsum
    out2.append(row)

# Convert to DataFrame
result_df = pd.DataFrame(out2)

# Reorder columns to put the improvement score first
cols = ['Model', 'ImprovementScore_unweighted'] + \
       [col for col in result_df.columns if col not in ['Model', 'ImprovementScore_unweighted']]
result_df = result_df[cols]

# Save to Excel
output_file = 'therese_unweighted_deltas_analysis.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    result_df.to_excel(writer, index=False, sheet_name='Unweighted Deltas')
    
    # Get the workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['Unweighted Deltas']
    
    # Set column width to make it more readable
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2) * 1.2
        worksheet.column_dimensions[column_letter].width = min(adjusted_width, 30)

print(f"Unweighted deltas analysis has been saved to {output_file}")
