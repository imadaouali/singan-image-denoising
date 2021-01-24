from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', default='zebra.png', required=True)
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Denoising')
    parser.add_argument('--ref_name', help='reference image name', required=True)
    parser.add_argument('--denoising_start_scale', help='denoising injection scale', type=int, required=True)
    parser.add_argument('--mode', help='task to be done', default='denoising')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        mode = opt.mode
        opt.mode = 'train'
        dir2trained_model = functions.generate_dir2save(opt)
        if (os.path.exists(dir2trained_model)):
            real = functions.read_image(opt)
            real = functions.adjust_scales2image(real, opt)
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            opt.mode = mode
        else:
            print('*** Train SinGAN for denoising ***')
            real = functions.read_image(opt)
            functions.adjust_scales2image(real, opt)
            train(opt, Gs, Zs, reals, NoiseAmp)
            SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
            opt.mode = mode
        real = functions.read_image(opt)
        real = functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        if (opt.denoising_start_scale < 1) | (opt.denoising_start_scale > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:
            ref = functions.read_image_dir('%s/%s' % (opt.ref_dir, opt.ref_name), opt)
            if ref.shape[3] != real.shape[3]:
                ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]

            N = len(reals) - 1
            n = opt.denoising_start_scale
            in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            plt.imsave('%s/start_scale=%d.png' % (dir2save, opt.denoising_start_scale), functions.convert_image_np(out.detach()), vmin=0, vmax=1)





