import cv2
import numpy as np
import os
import sys


class PersonsSplitter:
    """Generates individuals based on the mask with multiple merged individuals."""

    def __init__(self, data_dir, img_file, mask_file, n_persons=2):
        """
        :param data_dir: string, directory containing images
        :param img_file: string, image file
        :param mask_file: string, mask file
        :param n_persons: integer, number of individuals in the mask
        """
        self.data_dir = data_dir
        self.img_file = img_file
        self.mask_file = mask_file
        self.n_persons = n_persons
        self.img = cv2.imread(cv2.samples.findFile(os.path.join(self.data_dir, self.img_file)))
        self.mask = cv2.imread(cv2.samples.findFile(os.path.join(self.data_dir, self.mask_file)), cv2.IMREAD_GRAYSCALE)

        assert self.img is not None, f'Could not read image file <{self.img_file}>'
        assert self.mask is not None, f'Could not read mask file <{self.mask_file}>'

        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.persons_masks = self.create_individual_masks()  # numpy array of masks for each individual

    def create_individual_masks(self):
        """Creates an individual mask for each person"""
        mask_x_y = np.asarray((np.nonzero(self.mask > 0))).T

        # cv2.kmean only supports float
        retval, labels, centers = cv2.kmeans(
            np.float32(mask_x_y),
            self.n_persons,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.99),
            100,
            cv2.KMEANS_PP_CENTERS
        )

        persons_masks = np.zeros((self.n_persons, self.mask.shape[0], self.mask.shape[1]), dtype=np.uint8)

        for i in set(labels.ravel()):
            persons_masks[
                i,
                mask_x_y[:, 0][labels.ravel() == i],
                mask_x_y[:, 1][labels.ravel() == i]
            ] = 255

        return persons_masks

    def get_individual_person(self):
        """Unique person by mask (Problem 1)"""
        persons = np.zeros((self.n_persons, *self.img.shape), dtype=np.uint8)

        for i in range(self.n_persons):
            persons[i] = cv2.bitwise_and(self.img, self.img, mask=self.persons_masks[i])
            persons[i] = cv2.cvtColor(persons[i], cv2.COLOR_BGR2RGB)

        return persons

    def get_highlighted_persons(self):
        """Highlighted persons (Problem 2)"""
        np.random.seed(120)
        img_highlighted = np.copy(self.img)

        for i in range(self.n_persons):
            colored_mask = np.zeros(self.img.shape, dtype=np.uint8)
            ind = np.where(self.persons_masks[i] == 255)
            colored_mask[ind[0], ind[1], :] = np.random.randint(0, 255, 3)
            img_highlighted = cv2.add(img_highlighted, colored_mask)

        return cv2.cvtColor(img_highlighted, cv2.COLOR_BGR2RGB)

    def get_blured_persons(self):
        """Blured persons (Problem 3)"""
        img_blured = cv2.blur(self.img, (20, 20))
        img_persons_blured = np.copy(self.img)
        img_persons_blured[self.mask > 0] = img_blured[self.mask > 0]

        return cv2.cvtColor(img_persons_blured, cv2.COLOR_BGR2RGB)

    def save_img(self, mode='individual'):
        """
        Saves images to data_dir using one of the '.get_...' methods.

        - mode: str, one of the following: 'individual', 'highlighted', 'blured'
        """
        modes = ('individual', 'highlighted', 'blured')
        img_to_save = None

        if mode == modes[0]:
            img_to_save = self.get_individual_person()
        elif mode == modes[1]:
            img_to_save = self.get_highlighted_persons()
        else:
            img_to_save = self.get_blured_persons()

        is_multiple_imgs = len(img_to_save.shape) == 4

        for i in range(img_to_save.shape[0] if is_multiple_imgs else 1):
            log = cv2.imwrite(
                os.path.join(
                    self.data_dir,
                    self.img_file.split('.')[0] + '_' + modes[modes.index(mode)] + '_' + (str(i)) + '.png'
                ),
                img_to_save[i] if is_multiple_imgs else img_to_save
            )

            assert log, 'Could not save image'


if __name__ == '__main__':
    # Example of launching: python persons_from_masks.py data image.jpeg mask.png
    n_args_min = 4

    assert len(sys.argv) in (n_args_min, n_args_min + 1),\
        'Required number of arguments should be at least 3: <data dir> <img file> <mask file>'

    persons = PersonsSplitter(
        *(sys.argv[i + 1] for i in range(n_args_min - 1)),
        **(dict(n_persons=sys.argv[n_args_min]) if len(sys.argv) == n_args_min + 1 else {})
    )

    for mode in ('individual', 'highlighted', 'blured'):
        persons.save_img(mode=mode)
