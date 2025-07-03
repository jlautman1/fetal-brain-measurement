# fetal_normative.py

import numpy as np
import matplotlib.pyplot as plt
import os

# Regression model and SD for each measurement
def get_regression_stats(week, measure):
    if measure == 'CBD':
        mean = -19.82 + 5.12 * week - 0.036 * (week ** 2)
        sd = 5.0
    elif measure == 'BBD':
        mean = -20.43 + 5.45 * week - 0.037 * (week ** 2)
        sd = 5.0
    elif measure == 'TCD':
        mean = -4.42 + 2.17 * week - 0.016 * (week ** 2)
        sd = 3.0
    else:
        raise ValueError("Unknown measure type")
    return mean, sd

def get_normative_curve(measure):
    weeks = np.arange(22, 39)
    means = []
    sds = []
    for w in weeks:
        mean, sd = get_regression_stats(w, measure)
        means.append(mean)
        sds.append(sd)
    return weeks, np.array(means), np.array(sds)

def get_status(value, mean, sd):
    if value < mean - 2 * sd:
        return "Below Norm"
    elif value > mean + 2 * sd:
        return "Above Norm"
    else:
        return "Normal"

def predict_ga_from_measurement(value, measure):
    # Use quadratic formula: value = a + b*GA + c*GA^2
    # Solve for GA using numpy roots (ax^2 + bx + (c-value)=0)
    if measure == 'CBD':
        # -19.82 + 5.12*GA - 0.036*GA^2 = value
        a, b, c = -0.036, 5.12, -19.82 - value
    elif measure == 'BBD':
        a, b, c = -0.037, 5.45, -20.43 - value
    elif measure == 'TCD':
        a, b, c = -0.016, 2.17, -4.42 - value
    else:
        raise ValueError("Unknown measure")
    roots = np.roots([a, b, c])
    # Select the root that is in plausible GA range (22-38)
    week = None
    for r in roots:
        if np.isreal(r) and 20 <= r.real <= 40:
            week = r.real
            break
    return float(np.round(week, 1)) if week is not None else None

def plot_with_subject_point(measure, week, measured_value, outdir):
    weeks, means, sds = get_normative_curve(measure)
    upper = means + 2*sds
    lower = means - 2*sds
    
    # Create figure with professional styling
    plt.figure(figsize=(6, 4))
    plt.style.use('default')  # Reset to default style
    
    # Professional color scheme
    colors = {
        'primary': '#1f4e79',
        'secondary': '#4a90e2', 
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'light_blue': '#e8f4f8',
        'gray': '#6c757d'
    }
    
    # Plot the normative curve with professional styling
    plt.plot(weeks, means, label="Mean", color=colors['primary'], lw=3, alpha=0.9)
    plt.fill_between(weeks, lower, upper, 
                    color=colors['secondary'], alpha=0.2, 
                    label="±2 SD (Normal Range)")
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--', color='gray')
    
    # Determine point color based on status
    mean, sd = get_regression_stats(week, measure)
    status = get_status(measured_value, mean, sd)
    
    if status == "Normal":
        point_color = colors['success']
        edge_color = colors['success']
    elif status == "Below Norm":
        point_color = colors['warning']
        edge_color = colors['warning']
    else:
        point_color = colors['danger']
        edge_color = colors['danger']
    
    # Plot the subject point with professional styling
    plt.scatter([week], [measured_value], 
               color=point_color, 
               s=120, 
               edgecolor='white', 
               linewidths=2, 
               zorder=10,
               label='Subject Measurement')
    
    # Add a subtle annotation line
    plt.annotate(f'{measured_value:.1f}mm', 
                xy=(week, measured_value), 
                xytext=(week + 1, measured_value + 2),
                fontsize=10, 
                fontweight='bold',
                color=point_color,
                arrowprops=dict(arrowstyle='->', 
                              color=point_color, 
                              alpha=0.7,
                              lw=1.5))
    
    # Professional title and labels
    status_display = {"Below Norm": "Below Normal Range", 
                     "Above Norm": "Above Normal Range", 
                     "Normal": "Within Normal Range"}[status]
    
    plt.title(f'{measure} Normative Analysis\n{measured_value:.1f}mm at {week}w ({status_display})', 
             fontsize=14, fontweight='bold', 
             color=colors['primary'], pad=20)
    
    plt.xlabel('Gestational Age (weeks)', fontsize=12, fontweight='bold', color=colors['gray'])
    plt.ylabel(f'{measure} (mm)', fontsize=12, fontweight='bold', color=colors['gray'])
    
    # Professional styling for ticks and labels
    plt.xticks(fontsize=11, color=colors['gray'])
    plt.yticks(fontsize=11, color=colors['gray'])
    
    # Customize legend
    legend = plt.legend(frameon=True, loc='upper left', fontsize=10, 
                       fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(colors['gray'])
    legend.get_frame().set_alpha(0.9)
    
    # Set axis limits with some padding
    plt.xlim(weeks.min() - 0.5, weeks.max() + 0.5)
    y_range = upper.max() - lower.min()
    plt.ylim(lower.min() - 0.1*y_range, upper.max() + 0.1*y_range)
    
    # Professional spine styling
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color(colors['gray'])
        spine.set_linewidth(1)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fname = os.path.join(outdir, f"{measure.lower()}_norm.png")
    plt.savefig(fname, dpi=200, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    return fname, status

def normative_report_all(measured_dict, week, outdir):
    """
    measured_dict: dict, e.g. {'CBD': 90, 'BBD': 95, 'TCD': 42}
    week: gestational age in weeks (int or float)
    outdir: output directory to save graphs
    Returns: dict with status, plot paths, and predicted GA for each measurement
    """
    os.makedirs(outdir, exist_ok=True)
    results = {}
    for measure in ['CBD', 'BBD', 'TCD']:
        val = measured_dict[measure]
        fname, status = plot_with_subject_point(measure, week, val, outdir)
        mean, sd = get_regression_stats(week, measure)
        pred_ga = predict_ga_from_measurement(val, measure)
        results[measure] = {
            "value": val,
            "mean": np.round(mean, 2),
            "sd": sd,
            "status": status,
            "plot_path": fname,
            "predicted_ga": pred_ga
        }
    return results