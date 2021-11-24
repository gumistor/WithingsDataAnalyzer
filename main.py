import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as m_dates
import seaborn as sns
import os


sns.set(style='darkgrid')

fileList = os.listdir(r'data')

df_bp = pd.read_csv(os.path.join('.\\data', 'bp.csv'), sep=',', decimal=".")
df_wg = pd.read_csv(os.path.join('.\\data', 'weight.csv'), sep=',', decimal=".")

df_bp = df_bp.drop(labels=['Comments'], axis=1)
df_wg = df_wg.drop(labels=['Fat mass (kg)', 'Bone mass (kg)', 'Muscle mass (kg)', 'Hydration (kg)', 'Comments'], axis=1)


def align_datetime(df):
    df.loc[:, 'Date'] = pd.to_datetime(df.loc[:, 'Date'], format='%Y-%m-%d %H:%M:%S')
    # same measurement mean to one instance
    df = df.groupby(['Date']).mean().reset_index()
    # sort by date
    df = df.sort_values(by=['Date']).reset_index()
    # categorise by Time of day - Period
    df.loc[:, 'Period'] = (df.loc[:, 'Date'].dt.hour % 24 + 4) // 4
    df.loc[:, 'Period'].replace({
        1: 'Night',
        2: 'Morning',
        3: 'Morning',
        4: 'Noon',
        5: 'Evening',
        6: 'Night'}, inplace=True)
    return df


# prepare Date column
df_bp = align_datetime(df_bp)
df_wg = align_datetime(df_wg)

start_date = pd.to_datetime('01-01-2018 00:00:00')

df_bp = df_bp.loc[df_bp['Date'] > start_date]
df_wg = df_wg.loc[df_wg['Date'] > start_date]

# df_bp = df_bp.set_index('Date').between_time('05:00:00', '10:00:00').reset_index()
# df_wg = df_wg.set_index('Date').between_time('05:00:00', '10:00:00').reset_index()

df_bp_morning = df_bp.loc[df_bp['Period'] == 'Morning', :]
df_wg_morning = df_wg.loc[df_wg['Period'] == 'Morning', :]

# mean values calculation and plotting
df_bp_mean = df_bp_morning.set_index('Date').rolling('7D', center=False).mean()
df_wg_mean = df_wg_morning.set_index('Date').rolling('7D', center=False).mean()

# Plotting section
fig1, ax1 = plt.subplots(nrows=3, figsize=(21, 12), sharex='all')
fig1.canvas.manager.set_window_title("Time plots")
fig1.subplots_adjust(top=0.965,
                     bottom=0.08,
                     left=0.075,
                     right=0.895,
                     hspace=0.075,
                     wspace=0.185)

# Blood pressure and heart rate plotting
sns.lineplot(x='Date', y='Systolic', data=df_bp_morning, ax=ax1[0], label='Systolic', marker='.', linestyle='')
sns.lineplot(x='Date', y='Diastolic', data=df_bp_morning, ax=ax1[0], label='Diastolic', marker='.', linestyle='')
sns.lineplot(x='Date', y='Systolic', data=df_bp_mean, ax=ax1[0], label='Systolic mean', marker='', linestyle='-')
sns.lineplot(x='Date', y='Diastolic', data=df_bp_mean, ax=ax1[0], label='Diastolic mean', marker='', linestyle='-')
ax1[0].legend(bbox_to_anchor=(1, 1), loc="upper left")
ax1[0].set_ylabel('Blood pressure [mmHg]')

sns.lineplot(x='Date', y='Heart rate', data=df_bp_morning, ax=ax1[1], label='Heart rate', marker='.', linestyle='')
sns.lineplot(x='Date', y='Heart rate', data=df_bp_mean, ax=ax1[1], label='Heart rate mean', marker='', linestyle='-')
ax1[1].legend(bbox_to_anchor=(1, 1), loc="upper left")
ax1[1].set_ylabel('Heart rate [bps]')

# Weight ploting
sns.lineplot(x='Date', y='Weight (kg)', data=df_wg_morning, ax=ax1[2], label='Weight', marker='.', linestyle='')
sns.lineplot(x='Date', y='Weight (kg)', data=df_wg_mean, ax=ax1[2], label='Weight mean', marker='', linestyle='-')
ax1[2].legend(bbox_to_anchor=(1, 1), loc="upper left")
ax1[2].set_ylabel('Weight (kg)')

# show only year and month
date_form = m_dates.DateFormatter("%Y.%m")
ax1[2].set_xticklabels([], rotation=45, fontsize=10)
ax1[2].xaxis.set_major_locator(m_dates.MonthLocator(interval=1))
ax1[2].xaxis.set_major_formatter(date_form)
ax1[0].margins(x=0.002)   # TODO: to find more nice way to make a little space at the end of plot

for i in range(0, 3):
    ax1[i].axvspan(pd.to_datetime('2021-02-28 07:00:01'),
                   df_wg.iloc[-1, df_wg.columns.get_loc('Date')], facecolor='y', alpha=0.5)
    ax1[i].axvspan(pd.to_datetime('2021-10-25 20:00:01'),
                   df_wg.iloc[-1, df_wg.columns.get_loc('Date')], facecolor='y', alpha=0.4)

# merge two databases
df_merged = pd.merge_asof(df_bp_morning.set_index('Date'), df_wg.set_index('Date'),
                          on='Date', by='Period')  # tolerance=pd.Timedelta(12, 'h'), direction='nearest')

# remove columns after merging
df_merged = df_merged.drop(labels=['index_x', 'index_y', 'Period'], axis=1)

# generate correlation matrices
pd.plotting.scatter_matrix(df_merged, figsize=(21, 12))

# find correlation coefficient
df_merged_corr = df_merged.corr()

fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(21, 12))
fig3.canvas.manager.set_window_title("Correlations")
fig3.subplots_adjust(top=0.965,
                     bottom=0.08,
                     left=0.075,
                     right=0.895,
                     hspace=0.075,
                     wspace=0.185)

# plot correlation coefficient
sns.heatmap(df_merged_corr, mask=np.zeros_like(df_merged_corr, dtype=bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax3)

fig4, ax4 = plt.subplots(nrows=2, ncols=2, figsize=(21, 12))
fig4.canvas.manager.set_window_title("Histograms")
fig4.subplots_adjust(top=0.965,
                     bottom=0.08,
                     left=0.075,
                     right=0.895,
                     hspace=0.075,
                     wspace=0.185)

sys_min = df_bp_morning['Systolic'].astype(int).min()
sys_max = df_bp_morning['Systolic'].astype(int).max()
dia_min = df_bp_morning['Diastolic'].astype(int).min()
dia_max = df_bp_morning['Diastolic'].astype(int).max()

unique_periods = df_bp.loc[:, 'Period'].unique()

plot_index = 0
for period_selected in unique_periods:
    df_local = df_bp.loc[df_bp['Period'] == period_selected, :]
    ax4[plot_index % 2][plot_index // 2].hist2d(x=df_local.loc[:, 'Systolic'], y=df_local.loc[:, 'Diastolic'],
                                                bins=[np.linspace(sys_min, sys_max, sys_max-sys_min),
                                                np.linspace(dia_min, dia_max, dia_max-dia_min)])
    ax4[plot_index % 2][plot_index // 2].set_title(period_selected)
    plot_index += 1

# TODO: Try to find expected value... if possible
# sns.scatterplot(x='Systolic', y='Diastolic', data=df_bp, hue='Period')

plt.show()
# TODO: Description of colours - medicines taken
# TODO: Measurement time analysis, how often, which period, what time etc.
