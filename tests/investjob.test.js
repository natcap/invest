import React from 'react';
// import ReactTestUtils from 'react-dom/test-utils';
import renderer from 'react-test-renderer';
import { InvestJob } from '../src/InvestJob';


test('InvestJob snapshot', () => {
  const component = renderer.create(
    <InvestJob />
  );
  let tree = component.toJSON();
  expect(tree).toMatchSnapshot();
})

// test('Link changes the class when hovered', () => {
//   const component = renderer.create(
//     <Link page="http://www.facebook.com">Facebook</Link>,
//   );
//   let tree = component.toJSON();
//   expect(tree).toMatchSnapshot();

//   // manually trigger the callback
//   tree.props.onMouseEnter();
//   // re-rendering
//   tree = component.toJSON();
//   expect(tree).toMatchSnapshot();

//   // manually trigger the callback
//   tree.props.onMouseLeave();
//   // re-rendering
//   tree = component.toJSON();
//   expect(tree).toMatchSnapshot();
// });