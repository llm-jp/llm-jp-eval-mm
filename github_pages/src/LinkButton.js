import "./LinkButton.css";

interface LinkButtonProps {
  url: string;
  children?: React.ReactNode;
}

const LinkButton = ({ url, children }: LinkButtonProps) => {
  const handleClick = () => window.open(url, "_blank", "noopener noreferrer");

  return (
    <button className="link-button" onClick={handleClick}>
      {children}
    </button>
  );
};
export default LinkButton;
