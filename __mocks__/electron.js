// export default class Electron {
// 	const remote = {
// 		dialog: {
// 			showOpenDialog: jest.fn()
// 		}
// 	}
// }

// export const remote = {
// 	dialog: jest.fn().mockImplementation(win => {
// 		showOpenDialog: jest.fn(win => win)
// 	})
// }

export const remote = {
	dialog: {
		showOpenDialog: jest.fn()
	}
}