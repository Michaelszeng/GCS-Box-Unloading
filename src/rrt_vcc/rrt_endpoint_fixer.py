import pickle
import os

# TEST_SCENE = "3DOFFLIPPER"
# TEST_SCENE = "5DOFUR3"
# TEST_SCENE = "6DOFUR3"
# TEST_SCENE = "7DOFIIWA"
TEST_SCENE = "7DOFBINS"
# TEST_SCENE = "7DOF4SHELVES"
# TEST_SCENE = "14DOFIIWAS"
# TEST_SCENE = "15DOFALLEGRO"
# TEST_SCENE = "BOXUNLOADING"

pickle_file = f'testing_positions/{TEST_SCENE}_endpts.pkl'
# pickle_file = f'{TEST_SCENE}_endpts.pkl'

if os.path.exists(pickle_file):
    # Open the pickle file to load the data
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
        
        print(data)
        
        print(f"Num start pts: {len(data['start_pts'])}")
        print(f"Num end pts: {len(data['end_pts'])}")

    # if "start_pts" in data:
    #     data["start_pts"] = data["start_pts"][:-1]
        
    # if "end_pts" in data:
    #     data["end_pts"] = []

    # Save the modified data back to the pickle file
    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)
else:
    print(f"{pickle_file} does not exist.")
    
print("done.")

if data: 
    print(f"Num start pts: {len(data['start_pts'])}")
    print(f"Num end pts: {len(data['end_pts'])}")