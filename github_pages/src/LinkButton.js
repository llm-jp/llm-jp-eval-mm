import './LinkButton.css';

const LinkButton = ({ url, children }) => {
  const handleClick = () => window.open(url, '_blank', 'noopener noreferrer');

  return (
    <button className='link-button' onClick={handleClick}>
      {children}
    </button>
  );
};
export default LinkButton;
