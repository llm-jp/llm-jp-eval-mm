import React from 'react';
import './Figure.css';

const Figure = ({ image, altText, caption }) => {
  return (
    <figure className='figure'>
      <img src={image} alt={altText} className='figure-image' />
      <figcaption className='figure-caption'>{caption}</figcaption>
    </figure>
  );
};
export default Figure;
