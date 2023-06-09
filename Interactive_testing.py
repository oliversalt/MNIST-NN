import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import math 

data = np.load('MNIST\Github\weights_baises.npz')
W1, b1, W2, b2, W3, b3 = data['W1'], data['b1'], data['W2'], data['b2'], data['W3'], data['b3']

class ImageDrawer:
    def __init__(self, master):
        self.master = master
        self.canvas_size = 280  # size in pixels
        self.image_size = 28  # size in pixels
        self.pixel_size = self.canvas_size // self.image_size  # size of one pixel
        self.image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        self.canvas = tk.Canvas(master, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

    def draw(self, event):
        center_x, center_y = event.x // self.pixel_size, event.y // self.pixel_size
        
        # Define pen radius in pixels (can be a floating-point number)
        pen_radius = 1.1
        
        # Calculate the square of the pen radius (to avoid using sqrt in distance calculation)
        radius_sq = pen_radius ** 2
        
        # Determine the range of pixels to check around the cursor
        search_radius = math.ceil(pen_radius)
        
        # Increment the value of the pixel and its neighbors within the pen radius
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                x = center_x + dx
                y = center_y + dy
                
                # Skip if out of bounds
                if x < 0 or x >= self.image_size or y < 0 or y >= self.image_size:
                    continue
                
                # Check if the pixel is within the pen radius
                if dx ** 2 + dy ** 2 <= radius_sq:
                    # Increment the value of the pixel
                    self.image[y, x] = min(self.image[y, x] + 40, 255)
                    
                    # Convert the value to a grayscale color (255 is white, 0 is black)
                    intensity = 255 - self.image[y, x]
                    color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
                    
                    # Update the canvas
                    self.canvas.create_rectangle(
                        x * self.pixel_size, y * self.pixel_size,
                        (x + 1) * self.pixel_size, (y + 1) * self.pixel_size,
                        fill=color, outline=color
                    )

    def get_image(self):
        return self.image


root = tk.Tk()
drawer = ImageDrawer(root)
root.mainloop()

# Access the drawn image as a numpy array
image = drawer.get_image()
vector_784 = image.reshape(-1)
drawn_number = (vector_784 / 255) - 0.5


def ReLU(Z):
    return np.maximum(Z,0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def get_predictions(A3):
    return np.argmax(A3, 0)

def plot_random_number(image_array, prediction):
    # Reshape the array into a 28x28 image
    image = image_array.reshape(28, 28)
    inverted_image = 255 - image

    # Create a scatter plot of the inverted image
    x, y = np.meshgrid(range(28), range(28))
    plt.scatter(x, 27 - y, c=inverted_image, cmap='gray', marker='s')

    # Set labels and title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Guess: {prediction}")

    # Show the plot
    plt.show()

def predict_single_example(x, W1, b1, W2, b2, W3, b3):
    # Ensure the input is a 2D array of shape (1, 784)
    x = np.reshape(x, (1, -1))

    # Perform forward propagation
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, x.T) # note that x is transposed
    
    # Get the prediction by selecting the index of the maximum value in A3
    prediction = np.argmax(A3)

    plot_random_number(x, prediction)
    return prediction  # Return the single prediction value

random_index= np.random.randint(1,100)
predict_single_example(drawn_number, W1, b1, W2, b2, W3, b3)

