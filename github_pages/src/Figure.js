import React from "react";
import "./Figure.css";

interface FigureProps {
  image: string;
  altText: string;
  caption: React.ReactNode;
}
export type { FigureProps };

const Figure = ({ image, altText, caption }: FigureProps) => {
  return (
    <figure className="figure">
      <img src={image} alt={altText} className="figure-image" />
      <figcaption className="figure-caption">{caption}</figcaption>
    </figure>
  );
};
export default Figure;
