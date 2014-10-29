import get_data
import painting_sda

if __name__ == '__main__':
    root_path='../data/Paintings/two_class/'
    get_data.get_all_styles(root_path)
    train_fn, test_fn = get_data.images_to_numpy(root_path)
    (val_score, test_score) = painting_sda.test_sda(train_fn,test_fn)
    print val_score, test_score
