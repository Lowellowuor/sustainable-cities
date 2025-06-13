from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import pandas as pd
from clustering import preprocess_and_cluster # Import your clustering logic

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
# IMPORTANT: Change this secret key to a long, random string in production!
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_very_secure_random_fallback_for_dev_only_change_this_for_prod')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max upload size

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Homepage: Introduction to the project and its goals."""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """
    Upload page: Allows users to upload a CSV file for clustering.
    Handles file validation, saves the file, and retrieves analysis parameters.
    """
    if request.method == 'POST':
        # Get analysis parameters from the form
        try:
            num_clusters = int(request.form.get('num_clusters', 5)) # Default to 5 if not provided
            lat_column = request.form.get('lat_column', 'pickup_lat') # Default to 'pickup_lat'
            lon_column = request.form.get('lon_column', 'pickup_lon') # Default to 'pickup_lon'

            # Basic validation for num_clusters
            if not (2 <= num_clusters <= 20):
                flash('Number of clusters must be between 2 and 20.', 'error')
                return redirect(request.url)
            if not lat_column or not lon_column:
                flash('Latitude and Longitude column names cannot be empty.', 'error')
                return redirect(request.url)

        except ValueError:
            flash('Invalid number of clusters. Please enter an integer.', 'error')
            return redirect(request.url)

        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                flash('File successfully uploaded. Running analysis...', 'success')
                # Redirect to results page, passing parameters via URL query for analyze_results
                return redirect(url_for('analyze_results',
                                         filename=filename,
                                         num_clusters=num_clusters,
                                         lat_column=lat_column,
                                         lon_column=lon_column))
            except Exception as e:
                flash(f'Error saving file: {e}', 'error')
                return redirect(request.url)
        else:
            flash('Allowed file types are CSV', 'error')
            return redirect(request.url)
    return render_template('upload.html')

@app.route('/analyze/<filename>')
def analyze_results(filename):
    """
    Analysis page: Performs clustering on the uploaded file and displays results.
    Receives analysis parameters from URL query.
    """
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash('File not found.', 'error')
        return redirect(url_for('upload_file'))

    # Retrieve analysis parameters from query arguments
    num_clusters = int(request.args.get('num_clusters', 5))
    lat_column = request.args.get('lat_column', 'pickup_lat')
    lon_column = request.args.get('lon_column', 'pickup_lon')

    cluster_centers_df = pd.DataFrame()
    plot_images = {}
    raw_output = "" # Initialize raw output string
    metrics_data = {} # Initialize metrics dictionary
    error_message = None

    try:
        # Call preprocess_and_cluster which now returns all plot images, raw output, and metrics
        centers, plots_dict, raw_out_str, metrics_dict, err = preprocess_and_cluster(
            filepath,
            n_clusters=num_clusters,
            lat_col_name=lat_column,
            lon_col_name=lon_column
        )
        cluster_centers_df = centers
        plot_images = plots_dict
        raw_output = raw_out_str # Assign captured output
        metrics_data = metrics_dict # Assign captured metrics
        error_message = err
        if error_message:
            flash(error_message, 'error')
        else:
            flash('Analysis complete!', 'success')
    except ValueError as e:
        error_message = f"Data Error: {e}. Please ensure your CSV has the specified '{lat_column}' and '{lon_column}' columns, and valid coordinates within Nairobi."
        flash(error_message, 'error')
    except Exception as e:
        error_message = f"An unexpected error occurred during analysis: {e}"
        flash(error_message, 'error')

    # Optional: Clean up the uploaded file after processing (or keep for audit)
    # os.remove(filepath)

    return render_template('results.html',
                           filename=filename,
                           num_clusters=num_clusters,
                           lat_column=lat_column,
                           lon_column=lon_column,
                           cluster_centers=cluster_centers_df.to_html(classes='table table-striped'),
                           plot_images=plot_images,
                           raw_output=raw_output, # Pass raw output to template
                           metrics=metrics_data, # Pass metrics to template
                           error=error_message)

@app.route('/about')
def about():
    """About page: Explains SDG 11 and project goals."""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=False) # debug=True is for development, set to False in production
