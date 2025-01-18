import './Footer.css';

const Footer = () => {
  return (
    <footer className='footer'>
      <div className='footer-content'>
        <p>
          The source code for this website is licensed under the{' '}
          <a
            href='https://creativecommons.org/licenses/by-sa/4.0/'
            target='_blank'
            rel='noopener noreferrer'
          >
            Creative Commons Attribution-ShareAlike 4.0 International License
          </a>
          .
        </p>
        <p>
          This website is inspired by and references{' '}
          <a
            href='https://nerfies.github.io/'
            target='_blank'
            rel='noopener noreferrer'
          >
            Nerfies
          </a>{' '}
          and{' '}
          <a
            href='https://mmmu-benchmark.github.io/'
            target='_blank'
            rel='noopener noreferrer'
          >
            MMMU
          </a>
          and{' '}
          <a
            href='https://mmmu-japanese-benchmark.github.io/JMMMU/'
            target='_blank'
            rel='noopener noreferrer'
          >
            JMMMU
          </a>
          .
        </p>
        <p>
          Please also check out the{' '}
          <a
            href='https://github.com/speed1313/llm-jp-eval-mm.github.io'
            target='_blank'
            rel='noopener noreferrer'
          >
            GitHub repo
          </a>
          .
        </p>
      </div>
    </footer>
  );
};

export default Footer;
