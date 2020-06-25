import path from 'path';

export const remote = {
	dialog: {
		showOpenDialog: jest.fn(),
		showSaveDialog: jest.fn()
	},
	app: {
		getPath: jest.fn().mockImplementation(() => {
			return path.resolve('tests/data')
		})
	}
}