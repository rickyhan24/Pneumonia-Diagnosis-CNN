<!-- wp:heading -->
<h2>Pneumonia Diagnosis using CNN's</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p><strong>Introduction</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In the world of healthcare, one of the major issues that medical professionals face is the correct diagnosis of conditions and diseases of patients.&nbsp; Not being able to correctly diagnose a condition is a problem for both the patient and the doctor.&nbsp; The doctor is not benefitting the patient in the appropriate way if the doctor misdiagnoses the patient.&nbsp; This could lead to malpractice lawsuits and overall hurt the doctor’s business.&nbsp; The patient suffers by not receiving the proper treatment and risking greater harm to health by the condition that goes undetected; further, the patient undergoes unnecessary treatment and takes unnecessary medications, costing the patient time and money.&nbsp; &nbsp;&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>If we can correctly diagnose a patient’s condition, we have the potential to solve the above-mentioned problems.&nbsp; If we can produce deep learning models that can classify whether a patient has a condition or not, that can determine which particular condition the patient has, and that can determine the severity of the condition, then medical professionals will be able to use these models to better diagnose their patients.&nbsp; Accurate diagnosis can also be useful by allowing for timely treatment of a patient; being misdiagnosed can cause a delay in receiving the proper treatment.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In this paper, we will perform deep learning to a dataset representing the chest x-rays of pediatric patients from Guangzhou Women and Children’s Medical Center, Guangzhou.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>I would like to apply a convolutional neural network (CNN) and try to classify a patient as either having pneumonia or not having pneumonia.&nbsp; This is a binary classification problem. &nbsp;I would also like to apply CNN’s to classify a patient as either having bacterial pneumonia, viral pneumonia, or no pneumonia.&nbsp; This is a 3-class classification problem.</p>
<!-- /wp:paragraph -->

<!-- wp:image {"id":400,"sizeSlug":"large"} -->
<figure class="wp-block-image size-large"><img src="https://www.onlinemathtraining.com/wp-content/uploads/2020/04/pneumonia-1024x317.jpg" alt="" class="wp-image-400"/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>[The normal chest X-ray (left panel) depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia (middle) typically exhibits a focal lobar consolidation, in this case in the right upper lobe (white arrows), whereas viral pneumonia (right) manifests with a more diffuse “interstitial” pattern in both lungs. (Kermany et al, 2018)]</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Apparently, the bacterial pneumonia has areas of opaqueness that are more concentrated in one lobe whereas viral pneumonia has opaque areas more spread out on both lungs.&nbsp; The right lung is divided into three lobes, and the left lung is divided into two lobes.&nbsp; It’s certainly not obvious to me how to tell the difference.&nbsp; Hopefully, deep learning can help us tell the difference.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Data Preparation</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>The dataset can be found here: <a href="https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia#IM-0007-0001.jpeg">https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia#IM-0007-0001.jpeg</a></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>The data comes in two folders, one for the training set and one for the test set.&nbsp; The training set folder contains a folder of images for pneumonia cases and a folder of images for normal cases.&nbsp; The training set consists of 5216 images total.&nbsp; The test set folder contains a folder of images for pneumonia cases and a folder of images for normal cases.&nbsp; The test set consists of 624 images total, approximately 10.68% of the total set of images.&nbsp; Unlike the case in classical machine learning, we don’t have to worry about the various attributes of the dataset; in the case of convolutional neural networks, we just have a set of images.&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Convolutional Neural Networks</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>There is, however, some preparation of the images that is necessary before applying an artificial neural network.&nbsp; The images need to be prepared using convolutional layers in a process called convolution.&nbsp; There are several stages in this process—convolution operation, ReLU operation, pooling, and flattening; the end result is a vector that we can feed into an artificial neural network.&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Here is an image of a general CNN architecture:</p>
<!-- /wp:paragraph -->

<!-- wp:image {"id":406,"sizeSlug":"large"} -->
<figure class="wp-block-image size-large"><img loading="lazy" width="640" height="197" src="https://www.onlinemathtraining.com/wp-content/uploads/2020/04/640px-Typical_cnn.png" alt="" class="wp-image-406" srcset="https://www.onlinemathtraining.com/wp-content/uploads/2020/04/640px-Typical_cnn.png 640w, https://www.onlinemathtraining.com/wp-content/uploads/2020/04/640px-Typical_cnn-300x92.png 300w" sizes="(max-width: 640px) 100vw, 640px"></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>[By Aphex34 - Own work, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=45679374]</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>During the convolution operation, various feature detectors are applied to the image, creating a stack of feature maps—this stack of feature maps is called a convolutional layer.&nbsp; ReLU is applied to each feature map to enhance non-linearity.&nbsp; During the pooling stage, also known as subsampling, we apply max-pooling (or some other type of pooling) to each feature map, creating smaller feature maps that preserve the relevant features of the image.&nbsp; The resulting stack of pooled featured maps forms the pooling layer.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Once we get to the pooling layer, consisting of pooled feature maps, each pooled feature map is flattened into a vector and the resulting vectors are combined sequentially into one vector.&nbsp; The entries of this vector are fed into the input units of the artificial neural network.&nbsp; Thus, the entries of the flattened vector corresponding to one image are fed into the input units of the ANN.&nbsp; (This is in contrast to ANN’s used on a classical dataset where the <em>attributes</em> of a single instance are fed into the input units of the ANN).&nbsp; The artificial neural network is then trained on the training set and tested on the test set.&nbsp; Here is an image of a general ANN:</p>
<!-- /wp:paragraph -->

<!-- wp:image {"id":403,"width":357,"height":312,"sizeSlug":"large"} -->
<figure class="wp-block-image size-large is-resized"><img src="https://www.onlinemathtraining.com/wp-content/uploads/2020/04/neural-network-diagram.jpg" alt="" class="wp-image-403" width="357" height="312"/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>The ANN begins where it says ‘Fully connected’ in the diagram for the CNN architecture.&nbsp; As you can see, a convolutional neural network is the combination of convolution and an ANN.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Building a CNN with Python</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In order to build the CNN, we import Keras libraries and packages:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>The Sequential package is used to initialise the CNN.&nbsp; The Convolution2D package is used to create the convolutional layers.&nbsp; The MaxPooling2D package is used to created the pooled feature maps.&nbsp; The Flatten package is used to flatten the stack of pooled feature maps into one vector that can be fed into the ANN.&nbsp; The Dense package is used to add layers to the ANN.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Next, we initialize the CNN by creating an object of the Sequential class.&nbsp; This object we will call ‘classifier’:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Initialising the CNN
classifier = Sequential()
</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>We’re going to add one convolutional layer of 32 &nbsp;filter maps by applying 32 filters (feature detectors) of dimension 3 by 3 to the input image.&nbsp; We want our input images to have dimension 64 by 64 and treated as color images with 3 channels.&nbsp; We also apply ReLU to each feature map using the activation function ‘relu’:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Step 1 - Convolution
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation = 'relu'))
</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>Now that we have our feature maps in the convolutional layer, we apply max-pooling to each feature map using a 2 by 2 grid.</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Step 2 - Pooling
classifier.add(MaxPooling2D((2,2)))
</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>Now that we have a pooling layer consisting of pooled feature maps, we flatten each pooled feature map into a vector and combine all the resulting vectors sequentially into one giant vector.</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Step 3 - Flattening
classifier.add(Flatten())
</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>Next, we’re going to add our artificial neural network.&nbsp; First, we add a hidden layer of 128 units and use the activation function ‘relu’.&nbsp; Second, we add the output layer consisting of one output unit and use the sigmoid function as the activation function; we use one output unit because our output is binary (either normal or pneumonia).</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Step 4 - Full connection
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>Now, we need to compile the CNN.&nbsp; We’re going to use ‘adam’ as the optimizer in stochastic gradient descent, binary cross-entropy for the loss function, and accuracy as the performance metric.</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Compiling the CNN
classifier.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=&#91;'accuracy'])
</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>Our training set and test set combined has a total of 5840 images; so, we’re going to apply image augmentation to increase the size of our training set and test set while reducing overfitting.&nbsp; We then fit the CNN to our augmented training set and test it on our augmented test set:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'chest_xraybinary/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'chest_xraybinary/test',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        epochs=25,
        validation_data=test_set)
</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>After 25 epochs, I got an accuracy of 95% on the training set and 89% on the test set.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Evaluating, Improving, and Tuning the CNN</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Previously, we built a CNN with one convolutional layer and one hidden layer.&nbsp; This time, we’re going to add a second convolutional layer and see if it improves performance.&nbsp; We simply add the following code after step 2—pooling:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Adding a second convolutional layer
classifier.add(Convolution2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D((2,2)))
</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>After 25 epochs, we get an accuracy of 96% on the training set and 91.5% on the test set.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Next, in addition to having a second convolutional layer, we’re going to add a second hidden layer and see if it improves performance.&nbsp; To add a second hidden layer, we simply duplicate the code for adding one hidden layer:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Step 4 - Full connection
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>After 25 epochs, we get an accuracy of 96% on the training set and 91.5% on the test set.&nbsp; Adding a second hidden layer did not improve performance.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Distinguishing between Bacterial and Viral Pneumonia</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Not only do we want to distinguish between normal and pneumonia x-rays but we want to distinguish between the bacterial and viral pneumonia x-rays.&nbsp; To do this, we split up the folder containing pneumonia cases into two folders, one for bacteria cases and one for virus cases.&nbsp; Now, we have a three-class classification problem where the classes are normal, bacteria, and virus.&nbsp; Just as we used a CNN to solve the binary classification problem, we can use a CNN to solve the three-class classification problem.&nbsp; The code stays the same with a few exceptions.&nbsp; In the artificial neural network phase of the CNN, we change the number of output units from 1 to 3 and the output activation function from ‘sigmoid’ to ‘softmax’.&nbsp; When compiling the CNN, the loss function is changed from ‘binary_crossentropy’ to ‘categorical_crossentropy’.&nbsp; When fitting the CNN to the images, instead of using the folder ‘chest_xraybinary’, we use the folder ‘chest_xray’ which contains the training and test set folders that each have three folders corresponding to the three classes.&nbsp; The class_mode is changed from ‘binary’ to ‘categorical’.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>After 25 epochs, I got an accuracy of 80.64% on the training set and 83.33% on the test set.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>A second convolutional layer was added; and, after 25 epochs, I got an accuracy of 81.33% on the training set and 85.9% on the test set.&nbsp; This is a slight improvement in performance.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In addition to the second convolutional layer, a second hidden layer was added; and, after 25 epochs, I got an accuracy of 80.25% on the training set and 86.7% on the test set.&nbsp; This is a slight improvement in performance on the test set.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In addition to the second convolutional layer and the second hidden layer, I changed the number of feature detectors in the second convolutional layer from 32 to 64.&nbsp; After 25 epochs, I got an accuracy of 81% on the training set and 87% on the test set.&nbsp; This is a slight improvement in test set performance from the CNN with two convolutional layers and two hidden layers.&nbsp; However, it’s a decent improvement in test set performance from the CNN with one convolutional layer and one hidden layer.&nbsp; I then increased the dimensions of the images fed into the CNN from 64 by 64 to 128 by 128; this resulted in an accuracy of 80.85% on the training set and 85.74% on the test set, a worse performance than what we’ve reached so far.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>I brought the dimensions of the input images back down to 64 by 64 and added a third convolutional layer of 128 feature detectors.&nbsp; This resulted in an accuracy of 81.54% on the training set and 87.34% on the test set, a very small improvement on what we’ve got so far.&nbsp; Given the current settings, I ran the CNN for 200 epochs; the training set accuracy steadily increased to 96.32% while the training set accuracy fluctuated between the low 80’s and mid-80’s, ending at 86.54%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Keeping three convolutional layers, I added more hidden layers for a total of 10 hidden layers; after 25 epochs, I got a training set accuracy of 79.54% and a test set accuracy of 83.33%.&nbsp; So, adding more hidden layers didn’t lead to an improvement.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Conclusion</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In this paper, we applied convolutional neural networks to the binary classification problem of determining which of two classes—normal or pneumonia—a chest x-ray falls under.&nbsp; We found that a CNN with one convolutional layer and one hidden layer achieved an accuracy of 95% on the training set and an accuracy of 89% on the test set.&nbsp; We then added a second convolutional layer to the CNN and achieved an accuracy of 96% on the training set and an accuracy of 91.5% on the test set; this improved performance by a little bit.&nbsp; Next, in addition to having a second convolutional layer, we added a second hidden layer and achieved an accuracy of 96% on the training set and an accuracy of 91.5% on the test set; this did not improve performance.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>We also applied convolutional neural networks to the three-class classification problem of determining which of three classes—normal, bacterial, or viral—a chest x-ray falls under.&nbsp; We found that a CNN with one convolutional layer and one hidden layer achieved an accuracy of 80.64% on the training set and an accuracy of 83.33% on the test set.&nbsp; We then added a second convolutional layer to the CNN and achieved an accuracy of 81.33% on the training set and an accuracy of 85.9% on the test set, which was a slight improvement.&nbsp; Next, in addition to having a second convolutional layer, we added a second hidden layer and achieved an accuracy of 80.25% on the training set and an accuracy of 86.7% on the test set, which was a slight improvement in performance on the test set but a decline in improvement on the training set.&nbsp; In addition to the second convolutional layer and the second hidden layer, changing the number of feature detectors in the second convolutional layer from 32 to 64 resulted, after 25 epochs, in an accuracy of 81% on the training set and 87% on the test set.&nbsp; Adding a third convolutional layer of 128 feature detectors resulted in an accuracy of 81.54% on the training set and 87.34% on the test set.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Because we had a limited number of images in our total dataset, it was important to use image augmentation to provide more images to train our CNN; this lack of sufficient numbers of medical images seems to be a common problem for other medical issues besides pneumonia.&nbsp; Therefore, image augmentation promises to be a useful tool in any case in which there aren’t sufficient numbers of images.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Given that, in certain parts of the world, there may be a shortage of trained professionals who can read and interpret x-rays, there is the potential for automated diagnosis based on a patient’s x-ray.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In the case of classical machine learning techniques, sometimes we are able to identify particular attributes that are significant in determining the output of the machine learning model.&nbsp; In the case of convolutional neural networks, on the other hand, there is no set of attributes that we can identify as significant in determining the output of the CNN model; all we have are the images and their pixels.&nbsp; Further, it’s difficult to understand, in an intuitive way, how the CNN model is making its classifications.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>The CNN we’ve trained used chest x-rays of pediatric patients aged 1-5 from Guangzhou, China.&nbsp; Can our CNN be applied to children of other ages, to children outside of China, or even to adults?&nbsp; How might deep learning be used to detect the severity of a given case of pneumonia?&nbsp; These are open questions worth pursuing to further understand how image classification can be used to make medical diagnoses.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>The dataset can be found here:</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><a href="https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia#IM-0007-0001.jpeg">https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia#IM-0007-0001.jpeg</a></p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3>Acknowledgements</h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Data:&nbsp;<a href="https://data.mendeley.com/datasets/rscbjbr9sj/2" target="_blank" rel="noreferrer noopener">https://data.mendeley.com/datasets/rscbjbr9sj/2</a></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>License:&nbsp;<a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer noopener">CC BY 4.0</a></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Citation:&nbsp;<a href="http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5" target="_blank" rel="noreferrer noopener">http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5</a></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>References</strong><strong></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Kermany et al. Illustrative Examples of Chest X-Rays in Patients with Pneumonia.&nbsp;<em>Identifying Medical Diagnoses and Treatable</em></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><em>Diseases by Image-Based Deep Learning,</em>&nbsp;Cell 172, 1122-1131, 22 Feb. 2018, Elsevier Inc., <a href="https://doi.org/10.1016/j.cell.2018.02.010">https://doi.org/10.1016/j.cell.2018.02.010</a></p>
<!-- /wp:paragraph -->
