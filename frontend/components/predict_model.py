'''


# Python code to load and use the model

import pickle

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_new)
                        """)
                    elif format_type == "onnx":
                        st.code("""
# Python code to load and use the ONNX model
import onnxruntime as rt
import numpy as np

# Load the model
session = rt.InferenceSession('model.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Make predictions (ensure X_new is numpy array with correct shape)
predictions = session.run([output_name], {input_name: X_new.astype(np.float32)})[0]
                        """)
                    else:
                        st.code("""
# The JSON file contains the model parameters
# You'll need to recreate the model architecture and load these parameters
# Example for a simple linear regression:
import json
import numpy as np

with open('model.json', 'r') as f:
    params = json.load(f)

# Prediction function using the parameters
def predict(X, params):
    return np.dot(X, params['weights']) + params['bias']
                        """)

                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Error downloading model: {response.text}")
        else:
            st.error(f"Error fetching model information: {response.text}")

    except Exception as e:
        st.error(f"Error in download section: {str(e)}")
'''